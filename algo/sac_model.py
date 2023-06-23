import torch.nn as nn
import torch
from utils import torch_ext
from torch import distributions as pyd
import math
import torch.nn.functional as F
from utils.torch_ext import RunningMeanStd, TanhTransform, SquashedNormal


class BaseModel:
    def __init__(self, model_class):
        self.model_class = model_class  # 'sac'

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
            self.running_mean_std = RunningMeanStd(obs_shape)

    def norm_obs(self, observation):
        with torch.no_grad():
            return (
                self.running_mean_std(observation)
                if self.normalize_input
                else observation
            )


#########################


class NetworkBuilder:
    def __init__(self, **kwargs):
        pass

    class BaseNetwork(nn.Module):
        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

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
        ):
            return self._build_sequential_mlp(
                input_size,
                units,
                activation,
                dense_func,
                norm_func_name=None,
            )


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
                "norm_only_first_layer": self.norm_only_first_layer,
            }

            critic_mlp_args = {
                "input_size": obs_dim + action_dim,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
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
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        def _build_critic(self, output_dim, **mlp_args):
            return DoubleQCritic(output_dim, **mlp_args)

        def _build_actor(self, output_dim, log_std_bounds, **mlp_args):
            return DiagGaussianActor(output_dim, log_std_bounds, **mlp_args)

        def load(self, params):
            self.separate = params.get("separate", True)
            self.units = params["mlp"]["units"]
            self.activation = params["mlp"]["activation"]
            self.initializer = params["mlp"]["initializer"]
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

            assert self.has_space

            self.is_discrete = "discrete" in params["space"]
            self.is_continuous = "continuous" in params["space"]
            if self.is_continuous:
                self.space_config = params["space"]["continuous"]


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
