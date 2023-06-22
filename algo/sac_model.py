import torch.nn as nn
import torch
from utils import torch_ext
from torch import distributions as pyd
import math
import torch.nn.functional as F


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False, mask=None):
        if self.training:
            if mask is not None:
                mean, var = torch_ext.get_mean_std_with_masks(input, mask)
            else:
                mean = input.mean(self.axis)  # along channel axis
                var = input.var(self.axis)
            (
                self.running_mean,
                self.running_var,
                self.count,
            ) = self._update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size()[0],
            )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(
                    input
                )
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(
                    input
                )
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = (
                torch.sqrt(current_var.float() + self.epsilon) * y
                + current_mean.float()
            )
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(
                    current_var.float() + self.epsilon
                )
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


class RunningMeanStdObs(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        assert isinstance(insize, dict)
        super(RunningMeanStdObs, self).__init__()
        self.running_mean_std = nn.ModuleDict(
            {
                k: RunningMeanStd(v, epsilon, per_channel, norm_only)
                for k, v in insize.items()
            }
        )

    def forward(self, input, unnorm=False):
        res = {k: self.running_mean_std[k](v, unnorm) for k, v in input.items()}
        return res


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


class BaseModel:
    def __init__(self, model_class):
        self.model_class = model_class

    def is_rnn(self):
        return False

    def is_separate_critic(self):
        return False

    def build(self, config):
        obs_shape = config["input_shape"]
        normalize_value = config.get("normalize_value", False)
        normalize_input = config.get("normalize_input", False)
        value_size = config.get("value_size", 1)
        return self.Network(
            self.network_builder.build(self.model_class, **config),
            obs_shape=obs_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size,
        )


class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size

        if normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,))
        if normalize_input:
            if isinstance(obs_shape, dict):
                self.running_mean_std = RunningMeanStdObs(obs_shape)
            else:
                self.running_mean_std = RunningMeanStd(obs_shape)

    def norm_obs(self, observation):
        with torch.no_grad():
            return (
                self.running_mean_std(observation)
                if self.normalize_input
                else observation
            )

    def unnorm_value(self, value):
        with torch.no_grad():
            return (
                self.value_mean_std(value, unnorm=True)
                if self.normalize_value
                else value
            )


#########################
class D2RLNet(torch.nn.Module):
    def __init__(self, input_size, units, activations, norm_func_name=None):
        torch.nn.Module.__init__(self)
        self.activations = torch.nn.ModuleList(activations)
        self.linears = torch.nn.ModuleList([])
        self.norm_layers = torch.nn.ModuleList([])
        self.num_layers = len(units)
        last_size = input_size
        for i in range(self.num_layers):
            self.linears.append(torch.nn.Linear(last_size, units[i]))
            last_size = units[i] + input_size
            if norm_func_name == "layer_norm":
                self.norm_layers.append(torch.nn.LayerNorm(units[i]))
            elif norm_func_name == "batch_norm":
                self.norm_layers.append(torch.nn.BatchNorm1d(units[i]))
            else:
                self.norm_layers.append(torch.nn.Identity())

    def forward(self, input):
        x = self.linears[0](input)
        x = self.activations[0](x)
        x = self.norm_layers[0](x)
        for i in range(1, self.num_layers):
            x = torch.cat([x, input], dim=1)
            x = self.linears[i](x)
            x = self.norm_layers[i](x)
            x = self.activations[i](x)
        return x


def _create_initializer(func, **kwargs):
    return lambda v: func(v, **kwargs)


class NetworkBuilder:
    def __init__(self, **kwargs):
        pass

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class BaseNetwork(nn.Module):
        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def _calc_input_size(self, input_shape, cnn_layers=None):
            if cnn_layers is None:
                assert len(input_shape) == 1
                return input_shape[0]
            else:
                return (
                    nn.Sequential(*cnn_layers)(torch.rand(1, *(input_shape)))
                    .flatten(1)
                    .data.size(1)
                )

        def _noisy_dense(self, inputs, units):
            return layers.NoisyFactorizedLinear(inputs, units)

        def _build_rnn(self, name, input, units, layers):
            if name == "identity":
                return torch_ext.IdentityRNN(input, units)
            if name == "lstm":
                return LSTMWithDones(
                    input_size=input, hidden_size=units, num_layers=layers
                )
            if name == "gru":
                return GRUWithDones(
                    input_size=input, hidden_size=units, num_layers=layers
                )

        def _build_sequential_mlp(
            self,
            input_size,
            units,
            activation,
            dense_func,
            norm_only_first_layer=False,
            norm_func_name=None,
        ):
            print("build mlp:", input_size)
            in_size = input_size
            layers = []
            need_norm = True
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(nn.ReLU())

                if not need_norm:
                    continue
                if norm_only_first_layer and norm_func_name is not None:
                    need_norm = False
                if norm_func_name == "layer_norm":
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == "batch_norm":
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit

            return nn.Sequential(*layers)

        def _build_mlp(
            self,
            input_size,
            units,
            activation,
            dense_func,
            norm_only_first_layer=False,
            norm_func_name=None,
            d2rl=False,
        ):
            if d2rl:
                act_layers = [nn.ReLU() for i in range(len(units))]
                return D2RLNet(input_size, units, act_layers, norm_func_name)
            else:
                return self._build_sequential_mlp(
                    input_size,
                    units,
                    activation,
                    dense_func,
                    norm_func_name=None,
                )

        def _build_conv(self, ctype, **kwargs):
            print("conv_name:", ctype)

            if ctype == "conv2d":
                return self._build_cnn2d(**kwargs)
            if ctype == "coord_conv2d":
                return self._build_cnn2d(conv_func=torch_ext.CoordConv2d, **kwargs)
            if ctype == "conv1d":
                return self._build_cnn1d(**kwargs)

        def _build_cnn2d(
            self,
            input_shape,
            convs,
            activation,
            conv_func=torch.nn.Conv2d,
            norm_func_name=None,
        ):
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(
                    conv_func(
                        in_channels=in_channels,
                        out_channels=conv["filters"],
                        kernel_size=conv["kernel_size"],
                        stride=conv["strides"],
                        padding=conv["padding"],
                    )
                )
                conv_func = torch.nn.Conv2d
                act = nn.ReLU()
                layers.append(act)
                in_channels = conv["filters"]
                if norm_func_name == "layer_norm":
                    layers.append(torch_ext.LayerNorm2d(in_channels))
                elif norm_func_name == "batch_norm":
                    layers.append(torch.nn.BatchNorm2d(in_channels))
            return nn.Sequential(*layers)

        def _build_cnn1d(self, input_shape, convs, activation, norm_func_name=None):
            print("conv1d input shape:", input_shape)
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(
                    torch.nn.Conv1d(
                        in_channels,
                        conv["filters"],
                        conv["kernel_size"],
                        conv["strides"],
                        conv["padding"],
                    )
                )
                act = nn.ReLU()
                layers.append(act)
                in_channels = conv["filters"]
                if norm_func_name == "layer_norm":
                    layers.append(torch.nn.LayerNorm(in_channels))
                elif norm_func_name == "batch_norm":
                    layers.append(torch.nn.BatchNorm2d(in_channels))
            return nn.Sequential(*layers)


class DiagGaussianActor(NetworkBuilder.BaseNetwork):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, output_dim, log_std_bounds, **mlp_args):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        self.trunk = self._build_mlp(**mlp_args)
        last_layer = list(self.trunk.children())[-2].out_features
        self.trunk = nn.Sequential(
            *list(self.trunk.children()), nn.Linear(last_layer, output_dim)
        )

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        # log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # TODO: Refactor

        dist = SquashedNormal(mu, std)
        # Modify to only return mu and std
        return dist


class DoubleQCritic(NetworkBuilder.BaseNetwork):
    """Critic network, employes double Q-learning."""

    def __init__(self, output_dim, **mlp_args):
        super().__init__()

        self.Q1 = self._build_mlp(**mlp_args)
        last_layer = list(self.Q1.children())[-2].out_features
        self.Q1 = nn.Sequential(
            *list(self.Q1.children()), nn.Linear(last_layer, output_dim)
        )

        self.Q2 = self._build_mlp(**mlp_args)
        last_layer = list(self.Q2.children())[-2].out_features
        self.Q2 = nn.Sequential(
            *list(self.Q2.children()), nn.Linear(last_layer, output_dim)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2


class SACBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = SACBuilder.Network(self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            obs_dim = kwargs.pop("obs_dim")
            action_dim = kwargs.pop("action_dim")
            self.num_seqs = num_seqs = kwargs.pop("num_seqs", 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            mlp_input_shape = input_shape

            actor_mlp_args = {
                "input_size": obs_dim,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }

            critic_mlp_args = {
                "input_size": obs_dim + action_dim,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            print("Building Actor")
            self.actor = self._build_actor(
                2 * action_dim, self.log_std_bounds, **actor_mlp_args
            )

            if self.separate:
                print("Building Critic")
                self.critic = self._build_critic(1, **critic_mlp_args)
                print("Building Critic Target")
                self.critic_target = self._build_critic(1, **critic_mlp_args)
                self.critic_target.load_state_dict(self.critic.state_dict())

            mlp_init = nn.Identity()
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        def _build_critic(self, output_dim, **mlp_args):
            return DoubleQCritic(output_dim, **mlp_args)

        def _build_actor(self, output_dim, log_std_bounds, **mlp_args):
            return DiagGaussianActor(output_dim, log_std_bounds, **mlp_args)

        def forward(self, obs_dict):
            """TODO"""
            obs = obs_dict["obs"]
            mu, sigma = self.actor(obs)
            return mu, sigma

        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.separate = params.get("separate", True)
            self.units = params["mlp"]["units"]
            self.activation = params["mlp"]["activation"]
            self.initializer = params["mlp"]["initializer"]
            self.is_d2rl = params["mlp"].get("d2rl", False)
            self.norm_only_first_layer = params["mlp"].get(
                "norm_only_first_layer", False
            )
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)
            self.has_space = "space" in params
            self.value_shape = params.get("value_shape", 1)
            self.central_value = params.get("central_value", False)
            self.joint_obs_actions_config = params.get("joint_obs_actions", None)
            self.log_std_bounds = params.get("log_std_bounds", None)

            if self.has_space:
                self.is_discrete = "discrete" in params["space"]
                self.is_continuous = "continuous" in params["space"]
                if self.is_continuous:
                    self.space_config = params["space"]["continuous"]
                elif self.is_discrete:
                    self.space_config = params["space"]["discrete"]
            else:
                self.is_discrete = False
                self.is_continuous = False


########################


class ModelSACContinuous(BaseModel):
    def __init__(self, params):
        BaseModel.__init__(self, "sac")
        self.network_builder = SACBuilder()
        self.network_builder.load(params=params)

    class Network(BaseModelNetwork):
        def __init__(self, sac_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.sac_network = sac_network

        def critic(self, obs, action):
            return self.sac_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.sac_network.critic_target(obs, action)

        def actor(self, obs):
            return self.sac_network.actor(obs)

        def is_rnn(self):
            return False

        def forward(self, input_dict):
            is_train = input_dict.pop("is_train", True)
            mu, sigma = self.sac_network(input_dict)
            dist = SquashedNormal(mu, sigma)
            return dist
