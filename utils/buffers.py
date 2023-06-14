# replay buffer implementation from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
# rewritten to perform all the processing on GPU with pytorch Tensors

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import torch
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cuda",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: torch.Tensor) -> torch.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = torch.randint(
            0, upper_bound, size=(batch_size,), device=self.device
        )
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: torch.Tensor, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    # def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
    #     """
    #     Convert a numpy array to a PyTorch tensor.
    #     Note: it copies the data by default

    #     :param array:
    #     :param copy: Whether to copy or not the data
    #         (may be useful to avoid changing things be reference)
    #     :return:
    #     """
    #     if copy:
    #         return torch.tensor(array).to(self.device)
    #     return torch.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        env: Optional[VecNormalize] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(
        reward: torch.Tensor, env: Optional[VecNormalize] = None
    ) -> torch.Tensor:
        if env is not None:
            return env.normalize_reward(reward).astype(torch.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cuda",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        # assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = torch.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape,
            dtype=torch.float32,
            device=self.device,
        )

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = torch.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=torch.float32,
                device=self.device,
            )

        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.rewards = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device
        )
        self.dones = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device
        )
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = torch.zeros(
            (self.buffer_size, self.n_envs), dtype=torch.float32, device=self.device
        )

        if psutil is not None:
            total_memory_usage = (
                self.observations.storage().nbytes()
                + self.actions.storage().nbytes()
                + self.rewards.storage().nbytes()
                + self.dones.storage().nbytes()
            )

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.storage().nbytes()

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = obs.clone()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = next_obs.clone()
        else:
            self.next_observations[self.pos] = next_obs.clone()

        self.actions[self.pos] = action.clone()
        self.rewards[self.pos] = reward.clone()
        self.dones[self.pos] = done.clone()

        if self.handle_timeout_termination:
            self.timeouts[self.pos, :] = truncated[:]

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                torch.randint(
                    1, self.buffer_size, size=(batch_size,), device=self.device
                )
                + self.pos
            ) % self.buffer_size
        else:
            batch_inds = torch.randint(
                0, self.pos, size=(batch_size,), device=self.device
            )
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: torch.Tensor, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, 0, :], env
            )

        data = (
            self.swap_and_flatten(
                self._normalize_obs(self.observations[batch_inds, :], env)
            ),
            self.swap_and_flatten(self.actions[batch_inds, :]),
            self.swap_and_flatten(next_obs),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(data))
