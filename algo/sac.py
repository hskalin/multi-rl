# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


from env import env_map

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            env.num_obs + env.num_act,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.num_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, env.num_act)
        self.fc_logstd = nn.Linear(256, env.num_act)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.act_space.high - env.act_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.act_space.high + env.act_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SAC_agent:
    def __init__(self, args):
        self.args = args

        self.env = env_map[args.env_id](args)

        self.run_name = (
            f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        )
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        self.actor = Actor(self.env).to(self.device)
        self.qf1 = SoftQNetwork(self.env).to(self.device)
        self.qf2 = SoftQNetwork(self.env).to(self.device)
        self.qf1_target = SoftQNetwork(self.env).to(self.device)
        self.qf2_target = SoftQNetwork(self.env).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=args.policy_lr
        )

        # Automatic entropy tuning
        if args.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.act_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        self.rb = ReplayBuffer(
            args.buffer_size,
            self.env.obs_space,
            self.env.act_space,
            self.device,
            n_envs=self.args.num_envs,
            handle_timeout_termination=True,
        )

    def train(self):
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs = self.env.obs_buf.clone()

        for global_step in range(self.args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.args.learning_starts:
                actions = (
                    2
                    * torch.rand(
                        (self.args.num_envs, self.env.num_act),
                        device=self.device,
                        dtype=torch.float,
                    )
                    - 1
                )

            else:
                actions, _, _ = self.actor.get_action(obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            self.env.step(actions)
            next_obs, rewards, dones, episodeLen, episodeRet, truncated = (
                self.env.obs_buf.clone(),
                self.env.reward_buf.clone(),
                self.env.reset_buf.clone(),
                self.env.progress_buf.clone(),
                self.env.return_buf.clone(),
                self.env.truncated_buf.clone(),
            )
            self.env.reset()

            # next_obs, rewards, dones, infos = self.env.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            # only checking if first env is done to save computation
            if dones[0]:
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                # taking mean over all envs that are done at the
                # current timestep
                episodic_return = torch.mean(episodeRet[done_ids].float()).item()
                episodic_length = torch.mean(episodeLen[done_ids].float()).item()
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                self.writer.add_scalar(
                    "charts/episodic_return", episodic_return, global_step
                )
                self.writer.add_scalar(
                    "charts/episodic_length", episodic_length, global_step
                )

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs

            # TODO I think we do not need this since we are always recording
            # obs after the step

            # for idx, d in enumerate(dones):
            #     if d:
            #         real_next_obs[idx] = infos[idx]["terminal_observation"]

            self.rb.add(obs, real_next_obs, actions, rewards, dones, truncated)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs.clone()

            # ALGO LOGIC: training.
            if global_step > self.args.learning_starts:
                with torch.no_grad():
                    data = self.rb.sample(self.args.batch_size)
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                        data.next_observations
                    )
                    qf1_next_target = self.qf1_target(
                        data.next_observations, next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        data.next_observations, next_state_actions
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha * next_state_log_pi
                    )

                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)

                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if (
                    global_step % self.args.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    for _ in range(
                        self.args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if self.args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (
                                -self.log_alpha * (log_pi + self.target_entropy)
                            ).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_step % self.args.target_network_frequency == 0:
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    self.writer.add_scalar(
                        "losses/qf1_values", qf1_a_values.mean().item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/qf2_values", qf2_a_values.mean().item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/qf1_loss", qf1_loss.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/qf2_loss", qf2_loss.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step
                    )
                    self.writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
                    self.writer.add_scalar("losses/alpha", self.alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    if self.args.autotune:
                        self.writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )

        self.writer.close()
