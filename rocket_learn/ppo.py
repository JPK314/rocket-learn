import cProfile
import io
import os
import pstats
import sys
import time
from collections import defaultdict
from typing import Iterator

import numba
import numpy as np
import torch
import torch as th
from prettytable import PrettyTable
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator


class PPO:
    """
    Proximal Policy Optimization algorithm (PPO)

    :param rollout_generator: Function that will generate the rollouts
    :param agent: An ActorCriticAgent with actor as MultiHeadDiscretePolicy
    :param n_steps: The number of steps to run per update
    :param gamma: Discount factor
    :param batch_size: batch size to break experience data into for training
    :param epochs: Number of epoch when optimizing the loss
    :param minibatch_size: size to break batch sets into (helps combat VRAM issues)
    :param clip_range: PPO Clipping parameter for the value function
    :param ent_coef: Entropy coefficient for the loss calculation
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: optional clip_grad_norm value
    :param logger: wandb logger to store run results
    :param device: torch device
    :param zero_grads_with_none: 0 gradient with None instead of 0
    :param tick_skip_starts: a list of three tuples, from iteration 0, of (tick_skip, iteration_started, step_size)
    :param aux_heads_log_names: a list of strings to use for names for aux head related graphs

    Look here for info on zero_grads_with_none
    https://pytorch.org/docs/master/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad
    """

    def __init__(
        self,
        rollout_generator: BaseRolloutGenerator,
        agent: ActorCriticAgent,
        n_steps=4096,
        gamma=0.99,
        batch_size=512,
        epochs=10,
        # reuse=2,
        minibatch_size=None,
        clip_range=0.2,
        ent_coef=0.01,
        gae_lambda=0.95,
        vf_coef=1,
        max_grad_norm=0.5,
        logger=None,
        device="cuda",
        zero_grads_with_none=False,
        disable_gradient_logging=False,
        action_selection_dict=None,
        num_actions=0,
        tick_skip_starts=None,
        aux_heads_log_names=None,
        aux_weight_update_freq=5,
        aux_weight_learning_rate=1,
        keep_saved_aux_weights=True,
        aux_weight_max_change=None,
    ):
        self.tick_skip_starts = tick_skip_starts
        self.num_actions = num_actions
        self.action_selection_dict = action_selection_dict
        self.rollout_generator = rollout_generator

        # TODO let users choose their own agent
        # TODO move agent to rollout generator
        self.agent = agent.to(device)
        device_aux_heads = tuple(
            aux_head.to(device) for aux_head in self.agent.actor.aux_heads
        )
        total_aux_losses = sum(
            len(aux_head.get_weight()) for aux_head in self.agent.actor.aux_heads
        )
        self.aux_weight_update_freq = aux_weight_update_freq
        self.aux_weight_timer = 0
        self.hist_main_loss_grads = []
        self.hist_aux_loss_grads = [[] for _ in range(total_aux_losses)]
        self.aux_weight_learning_rate = aux_weight_learning_rate
        self.keep_saved_aux_weights = keep_saved_aux_weights
        self.aux_weight_max_change = aux_weight_max_change
        self.agent.actor.aux_heads = device_aux_heads
        if not aux_heads_log_names:
            aux_heads_log_names = [
                f"ppo/aux_head_{idx+1}"
                for idx, _ in enumerate(self.agent.actor.aux_heads)
            ]
        idx = len(aux_heads_log_names)
        while idx < total_aux_losses:
            aux_heads_log_names.append(f"ppo/aux_head_{idx}")
            idx += 1
        self.aux_heads_log_names = aux_heads_log_names
        self.device = device
        self.zero_grads_with_none = zero_grads_with_none
        self.frozen_iterations = 0

        self.starting_iteration = 0

        # hyperparameters
        self.epochs = epochs
        self.gamma = gamma
        # assert n_steps % batch_size == 0
        # self.reuse = reuse
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        assert self.batch_size % self.minibatch_size == 0
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.running_rew_mean = 0
        self.running_rew_var = 1
        self.running_rew_count = 1e-4

        self.total_steps = 0
        self.logger = logger
        if not disable_gradient_logging:
            self.logger.watch((self.agent.actor, self.agent.critic))
        self.timer = time.time_ns() // 1_000_000
        self.jit_tracer = None

    def update_reward_norm(self, rewards: np.ndarray) -> np.ndarray:
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = rewards.shape[0]

        delta = batch_mean - self.running_rew_mean
        tot_count = self.running_rew_count + batch_count

        new_mean = self.running_rew_mean + delta * batch_count / tot_count
        m_a = self.running_rew_var * self.running_rew_count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta)
            * self.running_rew_count
            * batch_count
            / (self.running_rew_count + batch_count)
        )
        new_var = m_2 / (self.running_rew_count + batch_count)

        new_count = batch_count + self.running_rew_count

        self.running_rew_mean = new_mean
        self.running_rew_var = new_var
        self.running_rew_count = new_count

        return (rewards - self.running_rew_mean) / np.sqrt(
            self.running_rew_var + 1e-8
        )  # TODO normalize before update?

    def run(self, iterations_per_save=10, save_dir=None, save_jit=False):
        """
        Generate rollout data and train
        :param iterations_per_save: number of iterations between checkpoint saves
        :param save_dir: where to save
        """
        if save_dir:
            current_run_dir = os.path.join(
                save_dir, self.logger.project + "_" + str(time.time())
            )
            os.makedirs(current_run_dir)
        elif iterations_per_save and not save_dir:
            print("Warning: no save directory specified.")
            print("Checkpoints will not be saved.")

        iteration = self.starting_iteration
        rollout_gen = self.rollout_generator.generate_rollouts()

        self.rollout_generator.update_parameters(
            self.agent.actor, self.agent.critic, iteration
        )

        while True:
            # pr = cProfile.Profile()
            # pr.enable()
            t0 = time.time()

            def _iter():
                size = 0
                print(f"Collecting rollouts ({iteration})...")
                while size < self.n_steps:
                    try:
                        rollout = next(rollout_gen)
                        if rollout.size() > 0:
                            size += rollout.size()
                            # progress.update(rollout.size())
                            yield rollout
                    except StopIteration:
                        return

            self.calculate(_iter(), iteration)
            iteration += 1

            if save_dir:
                self.save(
                    os.path.join(save_dir, self.logger.project + "_" + "latest"),
                    -1,
                    save_jit,
                )
                if iteration % iterations_per_save == 0:
                    self.save(current_run_dir, iteration, save_jit)  # noqa

            if self.frozen_iterations > 0:
                if self.frozen_iterations == 1:
                    print(" ** Unfreezing policy network **")

                    for param in self.agent.actor.parameters():
                        param.requires_grad = True

                self.frozen_iterations -= 1
            else:
                self.rollout_generator.update_parameters(self.agent.actor, iteration)

            # calculate years for graph
            if self.tick_skip_starts is not None:
                new_iteration = iteration
                years = 0
                for i in reversed(self.tick_skip_starts):
                    length = new_iteration - i[1]
                    years += length * i[2] / (3600 * 24 * 365 * (120 / i[0]))
                    new_iteration = i[1]
                self.logger.log({"ppo/years": years}, step=iteration, commit=False)

            self.total_steps += self.n_steps  # size
            t1 = time.time()
            self.logger.log(
                {
                    "ppo/steps_per_second": self.n_steps / (t1 - t0),
                    "ppo/total_timesteps": self.total_steps,
                }
            )
            # print(f"fps: {self.n_steps / (t1 - t0)}\ttotal steps: {self.total_steps}")
            seconds = self.total_steps * 8 / 120
            # def convert_time(seconds):
            #    minute, second = divmod(seconds, 60)
            #    hour, minute = divmod(minute, 60)
            #    day, hour = divmod(hour, 24)
            #    month, day = divmod(day, 30)
            #    year, month = divmod(month, 12)
            #    return (int(year), int(month), int(day), int(hour), int(minute), int(second))
            #
            # def convert_steps(total_steps):
            #    if total_steps < 1000:
            #        return total_steps, ""
            #    elif 1000 <= total_steps < 1000000:
            #        return total_steps / 1000, "K"
            #    elif 1000000 <= total_steps < 1000000000:
            #        return total_steps / 1000000, "M"
            #    else:
            #        return total_steps / 1000000000, "B"
            # def color_table_pink(table):
            #    for i in range(len(table._rows)):
            #        for j in range(len(table._field_names)):
            #            table._rows[i][j] = colored(table._rows[i][j], 'magenta') # Change The Color Here
            #    return table
            # times = convert_time(seconds)
            # steps, step_unit = convert_steps(self.total_steps)
            # x = PrettyTable()
            # x.field_names = ["Unit", "Value", "Steps"]
            # x.add_row(["", "", ""])
            # x.add_row([colored("Finesse Has Trained For", 'yellow', attrs=['bold']), "", ""])
            # x.add_row(["", "", ""])
            # x.add_row([colored("Years", 'magenta', attrs=['bold']), colored(times[0], 'magenta', attrs=['bold']), ""])
            # x.add_row([colored("Months", 'magenta', attrs=['bold']), colored(times[1], 'magenta', attrs=['bold']), ""])
            # x.add_row([colored("Days", 'magenta', attrs=['bold']), colored(times[2], 'magenta', attrs=['bold']), ""])
            # x.add_row([colored("Hours", 'magenta', attrs=['bold']), colored(times[3], 'magenta', attrs=['bold']), ""])
            # x.add_row([colored("Minutes", 'magenta', attrs=['bold']), colored(times[4], 'magenta', attrs=['bold']), ""])
            # x.add_row([colored("Seconds", 'magenta', attrs=['bold']), colored(times[5], 'magenta', attrs=['bold']), ""])
            # x.add_row([colored("Steps", 'magenta', attrs=['bold']), colored(format(steps, ".2f"), 'magenta', attrs=['bold']), step_unit])
            # x.add_row([colored("SPS", 'magenta', attrs=['bold']), colored(format(self.n_steps / (t1 - t0), ".2f"), 'magenta', attrs=['bold']), ""])
            # print(color_table_pink(x))
            # writer = tf.summary.create_file_writer("logs/PPO_0")
            # with writer.as_default():
            # tf.summary.scalar("ppo/steps_per_second", self.n_steps / (t1 - t0), step=self.total_steps)
            # tf.summary.scalar("ppo/total_timesteps", self.total_steps, step=self.total_steps)
            # tf.summary.scalar("time/time_elapsed", int(time_elapsed), step=iteration)

            # pr.disable()
            # s = io.StringIO()
            # sortby = pstats.SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.dump_stats(f"profile_{self.total_steps}")

    def set_logger(self, logger):
        self.logger = logger

    def evaluate_actions_aux_heads(self, observations, actions, aux_heads_labels):
        """
        Calculate Log Probability and Entropy of actions
        """
        body_out = self.agent.actor(observations)
        pred_aux_labels = self.agent.actor.get_aux_head_predictions(body_out)
        aux_heads_loss = self.agent.actor.get_aux_head_losses(
            aux_heads_labels, pred_aux_labels
        )
        dist = self.agent.actor.get_action_distribution(body_out)
        # indices = self.agent.get_action_indices(dists)

        log_prob = self.agent.actor.log_prob(dist, actions)
        entropy = self.agent.actor.entropy(dist, actions)

        entropy = -torch.mean(entropy)
        return log_prob, entropy, aux_heads_loss, dist

    @staticmethod
    @numba.njit
    def _calculate_advantages_numba(rewards, values, gamma, gae_lambda, truncated):
        advantages = np.zeros_like(rewards)
        # v_targets = np.zeros_like(rewards)
        dones = np.zeros_like(rewards)
        dones[-1] = 1.0 if not truncated else 0.0
        episode_starts = np.zeros_like(rewards)
        episode_starts[0] = 1.0
        last_values = values[-1]
        last_gae_lam = 0
        size = len(advantages)
        for step in range(size - 1, -1, -1):
            if step == size - 1:
                next_non_terminal = 1.0 - dones[-1].item()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1].item()
                next_values = values[step + 1]
            v_target = rewards[step] + gamma * next_values * next_non_terminal
            delta = v_target - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
            # v_targets[step] = v_target
        return advantages  # , v_targets

    def _get_flat_gradient(self, model):
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grad = p.grad.data.ravel()
            else:
                grad = th.zeros(p.shape, dtype=th.float, device=self.device).ravel()

            grads.append(grad)
        return th.cat(grads)

    def calculate(self, buffers: Iterator[ExperienceBuffer], iteration):
        """
        Calculate loss and update network
        """
        obs_tensors = []
        act_tensors = []
        # value_tensors = []
        log_prob_tensors = []
        # advantage_tensors = []
        returns_tensors = []

        rewards_tensors = []

        aux_heads_labels_tensors = tuple([] for _ in self.agent.actor.aux_heads)

        ep_rewards = []
        ep_steps = []
        action_count = np.asarray([0] * self.num_actions)
        action_changes = 0

        n = 0

        for buffer in buffers:  # Do discounts for each ExperienceBuffer individually
            if isinstance(buffer.observations[0], (tuple, list)):
                transposed = tuple(zip(*buffer.observations))
                obs_tensor = tuple(
                    torch.from_numpy(np.vstack(t)).float() for t in transposed
                )
            else:
                obs_tensor = th.from_numpy(np.vstack(buffer.observations)).float()

            with th.no_grad():
                if isinstance(obs_tensor, tuple):
                    x = tuple(o.to(self.device) for o in obs_tensor)
                else:
                    x = obs_tensor.to(self.device)
                values = (
                    self.agent.critic(x).detach().cpu().numpy().flatten()
                )  # No batching?

            actions = np.stack(buffer.actions)
            log_probs = np.stack(buffer.log_probs)
            rewards = np.stack(buffer.rewards)
            dones = np.stack(buffer.dones)
            aux_heads_labels = tuple(
                np.stack(aux_head_labels) for aux_head_labels in buffer.aux_labels
            )

            size = rewards.shape[0]

            advantages = self._calculate_advantages_numba(
                rewards, values, self.gamma, self.gae_lambda, dones[-1] == 2
            )

            returns = advantages + values
            if self.action_selection_dict is not None:
                flat_actions = actions[:, 0].flatten()
                unique, counts = np.unique(flat_actions, return_counts=True)
                for i, value in enumerate(unique):
                    action_count[value] += counts[i]
                action_changes += (np.diff(flat_actions) != 0).sum()

            obs_tensors.append(obs_tensor)
            act_tensors.append(th.from_numpy(actions))
            log_prob_tensors.append(th.from_numpy(log_probs))
            returns_tensors.append(th.from_numpy(returns))
            rewards_tensors.append(th.from_numpy(rewards))
            for aux_head_labels, aux_head_labels_tensors in zip(
                aux_heads_labels, aux_heads_labels_tensors
            ):
                aux_head_labels_tensors.append(th.from_numpy(aux_head_labels))

            ep_rewards.append(rewards.sum())
            ep_steps.append(size)
            n += 1
        ep_rewards = np.array(ep_rewards)
        ep_steps = np.array(ep_steps)
        ep_perstep_reward = ep_rewards.mean() / ep_steps.mean()

        total_steps = sum(ep_steps)
        self.logger.log(
            {
                "ppo/ep_reward_mean": ep_rewards.mean(),
                "ppo/ep_reward_std": ep_rewards.std(),
                "ppo/ep_len_mean": ep_steps.mean(),
                "ppo/mean_reward_per_step": ep_rewards.mean() / ep_steps.mean(),
            },
            step=iteration,
            commit=False,
        )

        # if ep_perstep_reward < 0:
        #    color = "red"
        # else:
        #    color = "green"
        # print(colored(f"RewardsPerStep: {ep_perstep_reward}", color))
        # mean_rewards = ep_rewards.mean()
        # if mean_rewards < 0:
        #    color = "red"
        # else:
        #    color = "green"
        # print(colored(f"Rewards: {mean_rewards}", color))
        # try:
        #    with open('new_rollouts.txt', 'a') as f:
        #        pass
        # except FileNotFoundError:
        #    with open('new_rollouts.txt', 'w') as f:
        #        pass
        # open('new_rollouts.txt', 'w').close()
        # with open('new_rollouts.txt', 'a') as f:
        #    f.write(f"RewardsPerStep: {ep_perstep_reward:.2f}\n")
        #    f.write(f"Rewards: {mean_rewards:.2f}\n")

        if self.action_selection_dict is not None:
            for k, v in self.action_selection_dict.items():
                count = action_count[k]
                name = "submodels/" + v
                ratio_used = count / total_steps
                self.logger.log({name: ratio_used}, step=iteration, commit=False)

        # print(f"std, mean rewards: {ep_rewards.std()}\t{ep_rewards.mean()}")

        if isinstance(obs_tensors[0], tuple):
            transposed = zip(*obs_tensors)
            obs_tensor = tuple(th.cat(t).float() for t in transposed)
        else:
            obs_tensor = th.cat(obs_tensors).float()
        act_tensor = th.cat(act_tensors)
        log_prob_tensor = th.cat(log_prob_tensors).float()
        # advantages_tensor = th.cat(advantage_tensors)
        returns_tensor = th.cat(returns_tensors).float()
        aux_heads_labels_tensor = tuple(
            th.cat(aux_head_labels_tensors).float()
            for aux_head_labels_tensors in aux_heads_labels_tensors
        )

        tot_loss = 0
        tot_policy_loss = 0
        tot_entropy_loss = 0
        tot_value_loss = 0
        tot_aux_heads_loss = [0 for _ in self.aux_heads_log_names]
        total_kl_div = 0
        tot_clipped = 0

        n = 0

        if self.jit_tracer is None:
            self.jit_tracer = obs_tensor[0].to(self.device)

        print("Training network...")

        if self.frozen_iterations > 0:
            print("Policy network frozen, only updating value network...")

        precompute = torch.cat(
            [param.view(-1) for param in self.agent.actor.parameters()]
        )
        aux_weights = self.agent.actor.get_aux_head_weights()
        t0 = time.perf_counter_ns()
        self.agent.optimizer.zero_grad(set_to_none=self.zero_grads_with_none)
        main_loss_total_grad = None
        aux_loss_total_grads = [None for _ in aux_weights]
        for e in range(self.epochs):
            # this is mostly pulled from sb3

            indices = torch.randperm(returns_tensor.shape[0])[: self.batch_size]
            if isinstance(obs_tensor, tuple):
                obs_batch = tuple(o[indices] for o in obs_tensor)
            else:
                obs_batch = obs_tensor[indices]
            act_batch = act_tensor[indices]
            log_prob_batch = log_prob_tensor[indices]
            # advantages_batch = advantages_tensor[indices]
            returns_batch = returns_tensor[indices]
            aux_heads_labels_batch = tuple(
                aux_head_labels_tensor[indices]
                for aux_head_labels_tensor in aux_heads_labels_tensor
            )
            aux_losses = [
                th.tensor(0, device=self.device, dtype=th.float)
                for _ in range(
                    sum(
                        len(aux_head.get_weight())
                        for aux_head in self.agent.actor.aux_heads
                    )
                )
            ]

            if isinstance(obs_tensor, tuple):
                obs = tuple(o.to(self.device) for o in obs_batch)
            else:
                obs = obs_batch.to(self.device)

            act = act_batch.to(self.device)
            # adv = advantages_batch[i:i + self.minibatch_size].to(self.device)
            ret = returns_batch.to(self.device)

            aux_heads_labels = tuple(
                aux_head_labels_batch.to(self.device)
                for aux_head_labels_batch in aux_heads_labels_batch
            )

            old_log_prob = log_prob_batch.to(self.device)

            # TODO optimization: use forward_actor_critic instead of separate in case shared, also use GPU
            try:
                (
                    log_prob,
                    entropy,
                    aux_heads_loss,
                    dist,
                ) = self.evaluate_actions_aux_heads(
                    obs, act, aux_heads_labels
                )  # Assuming obs and actions as input
            except RuntimeError as e:
                print("RuntimeError in evaluate_actions", e)
                (
                    log_prob,
                    entropy,
                    aux_heads_loss,
                    dist,
                ) = self.evaluate_actions_aux_heads(
                    obs, act, aux_heads_labels
                )  # Assuming obs and actions as input
            except ValueError as e:
                print("ValueError in evaluate_actions", e)
                continue

            ratio = torch.exp(log_prob - old_log_prob)

            try:
                values_pred = self.agent.critic(obs)
            except RuntimeError as e:
                print("RuntimeError in critic 2", e)
                values_pred = self.agent.critic(obs)

            values_pred = th.squeeze(values_pred)
            adv = ret - values_pred.detach()
            adv = (adv - th.mean(adv)) / (th.std(adv) + 1e-8)

            # clipped surrogate loss
            policy_loss_1 = adv * ratio
            policy_loss_2 = adv * th.clamp(
                ratio, 1 - self.clip_range, 1 + self.clip_range
            )
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # **If we want value clipping, add it here**
            value_loss = F.mse_loss(ret, values_pred)

            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = entropy

            for idx, aux_head_loss in enumerate(aux_heads_loss):
                aux_losses[idx] = aux_head_loss

            loss = (
                policy_loss
                + self.ent_coef * entropy_loss
                + self.vf_coef * value_loss
                + sum(aux_heads_loss)
            )

            if not torch.isfinite(loss).all():
                print("Non-finite loss, skipping", n)
                print("\tPolicy loss:", policy_loss)
                print("\tEntropy loss:", entropy_loss)
                print("\tValue loss:", value_loss)
                print("\tAux loss:", aux_heads_loss)
                print("\tTotal loss:", loss)
                print("\tRatio:", ratio)
                print("\tAdv:", adv)
                print("\tLog prob:", log_prob)
                print("\tOld log prob:", old_log_prob)
                print("\tEntropy:", entropy)
                print(
                    "\tActor has inf:",
                    any(not p.isfinite().all() for p in self.agent.actor.parameters()),
                )
                print(
                    "\tCritic has inf:",
                    any(not p.isfinite().all() for p in self.agent.critic.parameters()),
                )
                print("\tReward as inf:", not np.isfinite(ep_rewards).all())
                if isinstance(obs, tuple):
                    for j in range(len(obs)):
                        print(f"\tObs[{j}] has inf:", not obs[j].isfinite().all())
                else:
                    print("\tObs has inf:", not obs.isfinite().all())
                continue

            # Unbiased low variance KL div estimator from http://joschu.net/blog/kl-approx.html
            total_kl_div += th.mean((ratio - 1) - (log_prob - old_log_prob)).item()
            tot_loss += loss.item()
            tot_policy_loss += policy_loss.item()
            tot_entropy_loss += entropy_loss.item()
            tot_value_loss += value_loss.item()
            for idx, aux_head_loss in enumerate(aux_heads_loss):
                tot_aux_heads_loss[idx] += aux_head_loss.item()

            tot_clipped += th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
            n += 1
            # pb.update(self.minibatch_size)

            # Get loss gradients for aux weight update
            if self.frozen_iterations == 0:
                self.agent.optimizer.zero_grad(set_to_none=True)
                th.log(policy_loss).backward(retain_graph=True)
                main_loss_grad = self._get_flat_gradient(self.agent.actor) / self.epochs
                if main_loss_total_grad is None:
                    main_loss_total_grad = main_loss_grad
                else:
                    with th.no_grad():
                        main_loss_total_grad = main_loss_total_grad + main_loss_grad

                for idx, aux_loss in enumerate(aux_losses):
                    self.agent.optimizer.zero_grad(set_to_none=True)
                    th.log(aux_loss).backward(retain_graph=True)
                    aux_loss_grad = (
                        self._get_flat_gradient(self.agent.actor) / self.epochs
                    )
                    if aux_loss_total_grads[idx] is None:
                        aux_loss_total_grads[idx] = aux_loss_grad
                    else:
                        with th.no_grad():
                            aux_loss_total_grads[idx] = (
                                aux_loss_total_grads[idx] + aux_loss_grad
                            )

            # Set up computational graph for optimizer
            self.agent.optimizer.zero_grad(set_to_none=True)
            total_loss = (
                policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            )
            aux_weights = self.agent.actor.get_aux_head_weights()
            for aux_weight, aux_loss in zip(aux_weights, aux_losses):
                total_loss = total_loss + aux_weight * aux_loss
            total_loss.backward()
            # Clip grad norm
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)

            self.agent.optimizer.step()

        self.hist_main_loss_grads.append(main_loss_total_grad)
        for hist_aux_loss_grad, aux_loss_total_grad in zip(
            self.hist_aux_loss_grads, aux_loss_total_grads
        ):
            hist_aux_loss_grad.append(aux_loss_total_grad)

        # Perform aux weight update
        self.aux_weight_timer += 1
        with th.no_grad():
            if self.aux_weight_timer == self.aux_weight_update_freq:
                self.aux_weight_timer = 0
                update_vec = th.tensor(
                    [0] * len(self.hist_aux_loss_grads),
                    device=self.device,
                    dtype=th.float,
                )
                for update_vec_idx, hist_aux_loss_grad in enumerate(
                    self.hist_aux_loss_grads
                ):
                    for hist_idx, aux_loss_grad in enumerate(hist_aux_loss_grad):
                        update_vec[update_vec_idx] += (
                            th.dot(
                                aux_loss_grad,
                                self.hist_main_loss_grads[hist_idx],
                            )
                            / self.aux_weight_update_freq
                        )

                scaled_update_vec = self.aux_weight_learning_rate * update_vec
                if self.aux_weight_max_change is not None:
                    abs_scaled_update_vec = th.abs(scaled_update_vec)
                    max_ind = th.argmax(abs_scaled_update_vec)
                    max_val = abs_scaled_update_vec[max_ind]
                    if max_val > self.aux_weight_max_change:
                        scaled_update_vec = (
                            scaled_update_vec / max_val * self.aux_weight_max_change
                        )

                self.agent.actor.update_aux_head_weights(scaled_update_vec)
                self.hist_main_loss_grads = []
                self.hist_aux_loss_grads = [
                    [] for _ in range(len(self.hist_aux_loss_grads))
                ]
        t1 = time.perf_counter_ns()

        assert n > 0

        postcompute = torch.cat(
            [param.view(-1) for param in self.agent.actor.parameters()]
        )
        logdict = {
            "ppo/loss": tot_loss / n,
            "ppo/policy_loss": tot_policy_loss / n,
            "ppo/entropy_loss": tot_entropy_loss / n,
            "ppo/value_loss": tot_value_loss / n,
            "ppo/mean_kl": total_kl_div / n,
            "ppo/clip_fraction": tot_clipped / n,
            "ppo/epoch_time": (t1 - t0) / (1e6 * self.epochs),
            "ppo/update_magnitude": th.dist(precompute, postcompute, p=2),
        }
        for idx, aux_head_log_name in enumerate(self.aux_heads_log_names):
            logdict[aux_head_log_name] = tot_aux_heads_loss[idx]
            logdict[f"{aux_head_log_name}_weight"] = aux_weights[idx]

        # print(logdict)
        self.logger.log(
            logdict, step=iteration, commit=False
        )  # Is committed after when calculating fps

    def load(self, load_location, continue_iterations=True):
        """
        load the model weights, optimizer values, and metadata
        :param load_location: checkpoint folder to read
        :param continue_iterations: keep the same training steps
        """

        checkpoint = torch.load(load_location)
        initial_weights = self.agent.actor.get_aux_head_weights()
        initial_head_weight_maxes = []
        for aux_head in self.agent.actor.aux_heads:
            initial_head_weight_maxes.append(aux_head.max_weight_tensor)
        self.agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.agent.actor.aux_heads = checkpoint["aux_heads"]
        self.agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if not self.keep_saved_aux_weights:
            loaded_weights = self.agent.actor.get_aux_head_weights()
            self.agent.actor.update_aux_head_weights(initial_weights - loaded_weights)
        for initial_head_weight_max, aux_head in zip(
            initial_head_weight_maxes, self.agent.actor.aux_heads
        ):
            aux_head.max_weight_tensor = initial_head_weight_max
        if continue_iterations:
            self.starting_iteration = checkpoint["epoch"]
            self.total_steps = checkpoint["total_steps"]
            print("Continuing training at iteration " + str(self.starting_iteration))

    def save(self, save_location, current_step, save_actor_jit=False):
        """
        Save the model weights, optimizer values, and metadata
        :param save_location: where to save
        :param current_step: the current iteration when saved. Use to later continue training
        :param save_actor_jit: save the policy network as a torch jit file for rlbot use
        """

        version_str = str(self.logger.project) + "_" + str(current_step)
        version_dir = save_location + "\\" + version_str

        os.makedirs(version_dir, exist_ok=current_step == -1)

        torch.save(
            {
                "epoch": current_step,
                "total_steps": self.total_steps,
                "actor_state_dict": self.agent.actor.state_dict(),
                "aux_heads": tuple(self.agent.actor.aux_heads),
                "critic_state_dict": self.agent.critic.state_dict(),
                "optimizer_state_dict": self.agent.optimizer.state_dict(),
                # TODO save/load reward normalization mean, std, count
            },
            version_dir + "\\checkpoint.pt",
        )

        if save_actor_jit:
            traced_actor = th.jit.trace(self.agent.actor, self.jit_tracer)
            torch.jit.save(traced_actor, version_dir + "\\jit_policy.jit")

    def freeze_policy(self, frozen_iterations=100):
        """
        Freeze policy network to allow value network to settle. Useful with pretrained policy networks.

        Note that network weights will not be transmitted when frozen.

        :param frozen_iterations: how many iterations the policy update will remain unchanged
        """

        print("-------------------------------------------------------------")
        print("Policy Weights frozen for " + str(frozen_iterations) + " iterations")
        print("-------------------------------------------------------------")

        self.frozen_iterations = frozen_iterations

        for param in self.agent.actor.parameters():
            param.requires_grad = False
