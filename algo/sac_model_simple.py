import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import torch_ext

###### builders #######################


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

        dist = torch_ext.SquashedNormal(mu, std)
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


######################################


class Network(NetworkBuilder.BaseNetwork):
    def __init__(self, params, net_config):
        actions_num = net_config["actions_num"]
        input_shape = net_config["input_shape"]
        obs_dim = net_config["obs_dim"]
        action_dim = net_config["action_dim"]
        self.num_seqs = net_config.get("num_seqs", 1)

        self.normalize_input = net_config.get("normalize_value", False)
        self.normalize_value = net_config.get("normalize_value", False)

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

        if self.normalize_value:
            self.value_mean_std = torch_ext.RunningMeanStd(
                (net_config.get("value_size", 1),)
            )
        if self.normalize_input:
            self.running_mean_std = torch_ext.RunningMeanStd(input_shape)

    def _build_critic(self, output_dim, **mlp_args):
        return DoubleQCritic(output_dim, **mlp_args)

    def _build_actor(self, output_dim, log_std_bounds, **mlp_args):
        return DiagGaussianActor(output_dim, log_std_bounds, **mlp_args)

    def norm_obs(self, observation):
        with torch.no_grad():
            return (
                self.running_mean_std(observation)
                if self.normalize_input
                else observation
            )

    def load(self, params):
        self.separate = params.get("separate", True)
        self.units = params["mlp"]["units"]
        self.activation = params["mlp"]["activation"]
        self.initializer = params["mlp"]["initializer"]
        self.norm_only_first_layer = params["mlp"].get("norm_only_first_layer", False)
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
