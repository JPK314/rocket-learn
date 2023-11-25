import time
from typing import List

import numpy as np
import torch
from rlgym.gym import Gym
from rlgym.utils.reward_functions.common_rewards import ConstantReward
from rlgym.utils.state_setters import DefaultState
from rocketsimvisualizer import VisualizerThread
from termcolor import colored
from tqdm import tqdm

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter
from rocket_learn.utils.truncated_condition import (
    TerminalToTruncatedWrapper,
    TruncatedCondition,
)


def get_feature_importance(policy, obs):
    n_linspace = 100
    obs_ranges = []
    with open("obs_range.txt", "r") as f:
        for line in f:
            min_val, max_val = [float(v) for v in line.split(",")]
            obs_ranges.append(np.linspace(min_val, max_val, n_linspace))
    p0_obs = obs[0]
    body_out = policy(p0_obs)
    body_out_cuda = body_out.to(torch.device("cuda:0"))
    aux_head_preds = policy.get_aux_head_predictions(body_out_cuda)
    team_pred = aux_head_preds[0][0].item()
    modded_obs = p0_obs.copy()
    batch_mod_obs = np.repeat(modded_obs[np.newaxis, :], n_linspace, axis=0)
    pred_ranges = []
    for idx, obs_range in enumerate(obs_ranges):
        batch_mod_obs[:, idx] = obs_range
        batch_mod_body_out = policy(batch_mod_obs)
        batch_mod_body_out_cuda = batch_mod_body_out.to(torch.device("cuda:0"))
        batch_mod_aux_head_preds = policy.get_aux_head_predictions(
            batch_mod_body_out_cuda
        )
        batch_mod_team_pred = batch_mod_aux_head_preds[0]
        pred_ranges.append(
            (
                torch.min(batch_mod_team_pred).item(),
                torch.max(batch_mod_team_pred).item(),
            )
        )
    pred_ranges_sorted = list(enumerate(pred_ranges))
    pred_ranges_sorted.sort(key=lambda x: x[1][1] - x[1][0], reverse=True)
    print("\n".join(list(f"{idx}, {r[1]-r[0]}" for (idx, r) in pred_ranges_sorted)))
    print("oh boy")


def get_feature_importance2(env, policy, obs, pct_max_range, step):
    n_linspace = 100
    obs_rangess = [[] for _ in obs]
    file_line = 0
    with open("obs_range.txt", "r") as f:
        for line in f:
            min_val, max_val = [float(v) for v in line.split(",")]
            val_range = pct_max_range * (max_val - min_val)
            for idx, o in enumerate(obs):
                obs_rangess[idx].append(
                    np.linspace(
                        max(min_val, o[file_line] - val_range),
                        min(max_val, o[file_line] + val_range),
                        n_linspace,
                    )
                )
            file_line += 1
    body_out = policy(obs)
    dist = policy.get_action_distribution(body_out)
    base_vals = torch.sum(
        dist.probs[:, :, env._match._action_parser._jump_actions], axis=2
    )
    modded_obs = obs.copy()
    batch_mod_obs = np.repeat(modded_obs[np.newaxis, :], n_linspace, axis=0)
    pred_rangess = []
    for idx1, obs_ranges in enumerate(obs_rangess):
        pred_rangess.append([])
        for idx2, obs_range in enumerate(obs_ranges):
            batch_mod_obs[:, idx1, idx2] = obs_range
            batch_mod_body_out = policy(batch_mod_obs[:, idx1, :])
            batch_mod_dist = policy.get_action_distribution(batch_mod_body_out)
            batch_mod_jump_pref = torch.sum(
                batch_mod_dist.probs[:, 0, env._match._action_parser._jump_actions],
                axis=1,
            )
            pred_rangess[-1].append(
                (
                    torch.min(batch_mod_jump_pref).item(),
                    torch.max(batch_mod_jump_pref).item(),
                )
            )
    pred_rangess_sorted = [list(enumerate(pred_ranges)) for pred_ranges in pred_rangess]
    for pred_ranges_sorted in pred_rangess_sorted:
        pred_ranges_sorted.sort(key=lambda x: x[1][1] - x[1][0], reverse=True)
    with open("feature_importance_steps.txt", "a") as f:
        f.write(f"--------------STEP {step}--------------\n")
        for idx1 in range(20):
            f.write(
                "|".join(
                    [
                        f"{pred_ranges_sorted[idx1][0]}, {pred_ranges_sorted[idx1][1][1] - pred_ranges_sorted[idx1][1][0]}"
                        for pred_ranges_sorted in pred_rangess_sorted
                    ]
                )
                + "\n"
            )
        f.write("\n\n")


def generate_episode(env: Gym, policies, versions, eval_setter=DefaultState(), evaluate=False, scoreboard=None, progress=True, v: VisualizerThread = None) -> (List[ExperienceBuffer], int):  # type: ignore
    """
    create experience buffer data by interacting with the environment(s)
    """
    if progress:
        progress = tqdm(unit=" steps")
    else:
        progress = None

    # Change setup temporarily to play a normal game (approximately)
    if evaluate:
        # tools is an optional dependency
        from rlgym_tools.extra_terminals.game_condition import GameCondition

        terminals = env._match._terminal_conditions  # noqa
        reward = env._match._reward_fn  # noqa
        game_condition = GameCondition(
            tick_skip=8, max_overtime_seconds=300, max_no_touch_seconds=30  # noqa
        )  # noqa
        env._match._terminal_conditions = [game_condition]  # noqa
        if isinstance(env._match._state_setter, DynamicGMSetter):  # noqa
            state_setter = env._match._state_setter.setter  # noqa
            env._match._state_setter.setter = eval_setter  # noqa
            env.update_settings(boost_consumption=1)  # remove infinite boost
        else:
            state_setter = env._match._state_setter  # noqa
            env._match._state_setter = eval_setter  # noqa
            env.update_settings(boost_consumption=1)  # remove infinite boost

        env._match._reward_fn = ConstantReward()  # noqa Save some cpu cycles

    if scoreboard is not None:
        random_resets = scoreboard.random_resets
        scoreboard.random_resets = not evaluate
    observations, info = env.reset(return_info=True)
    # v.visualizer.update()
    result = 0

    last_state = info["state"]  # game_state for obs_building of other agents
    distinct_non_pretrained_versions_set = set(
        [
            v
            for idx, v in enumerate(versions)
            if not isinstance(policies[idx], HardcodedAgent)
        ]
    )
    policy_version_idx_dict = {}
    for version in distinct_non_pretrained_versions_set:
        policy_version_idx_dict[version] = [
            idx for idx, v in enumerate(versions) if v == version
        ]
    pretrained_idxs = [
        idx
        for idx, v in enumerate(versions)
        if isinstance(policies[idx], HardcodedAgent)
    ]

    latest_policy_indices = [
        0 if isinstance(p, HardcodedAgent) else 1 for p in policies
    ]
    # rollouts for all latest_policies
    rollouts = [
        ExperienceBuffer(infos=[info]) for _ in range(sum(latest_policy_indices))
    ]

    trajectory_states = []
    # body_outs = [[] for _ in range(sum(latest_policy_indices))]
    # step_outs_file = open("step_outs.txt", "w")
    # first_jumper_file = open("first_jumper.txt", "a")
    # someone_has_jumped = False
    # jumpers = []
    # t0 = time.time()
    b = o = 0
    with torch.no_grad():
        while True:
            # t1 = time.time()
            # time.sleep(max(0, 8 / 120 - (t1 - t0)))
            # t0 = t1
            # all_indices = []
            # all_actions = []
            # all_log_probs = []
            all_indices = [None] * len(policies)
            all_actions = [None] * len(policies)
            all_log_probs = [None] * len(policies)

            # if observation isn't a list, make it one so we don't iterate over the observation directly
            if not isinstance(observations, list):
                observations = [observations]

            # get action indices, actions, and log probs for non pretrained agents
            for idxs in policy_version_idx_dict.values():
                policy = policies[idxs[0]]
                if isinstance(observations[idxs[0]], tuple):
                    obs = tuple(
                        np.concatenate(
                            [
                                obs[i]
                                for idx, obs in enumerate(observations)
                                if idx in idxs
                            ],
                            axis=0,
                        )
                        for i in range(len(observations[idxs[0]]))
                    )
                else:
                    obs = np.concatenate(
                        [obs for idx, obs in enumerate(observations) if idx in idxs],
                        axis=0,
                    )
                # get_feature_importance2(env, policy, obs, 0.1, progress.n)
                body_out = policy(obs)
                # body_out_cuda = body_out.to(torch.device("cuda:0"))
                # aux_head_preds = policy.get_aux_head_predictions(body_out_cuda)
                # print(progress.n)
                # print(aux_head_preds[0].detach().cpu().numpy())
                # step_outs_file.write("\n" + str(progress.n) + " steps")
                # step_outs_file.write(
                #     "\n" + repr(aux_head_preds[0].detach().cpu().numpy())
                # )
                dist = policy.get_action_distribution(body_out)
                action_indices = policy.sample_action(dist)
                # step_outs_file.write("\n" + repr(action_indices))
                # step_outs_file.flush()
                # time.sleep(0.05)
                log_probs = policy.log_prob(dist, action_indices)
                action_indices_list = list(action_indices.numpy())
                log_probs_list = list(log_probs.numpy())
                # if progress.n > 114:
                #     get_feature_importance2(env, policy, obs, 0.1, progress.n)
                for i, idx in enumerate(idxs):
                    all_indices[idx] = action_indices_list[i]
                    all_log_probs[idx] = log_probs_list[i]
                    all_actions[idx] = policy.env_compatible(action_indices[i])
                # if not someone_has_jumped:
                #     for i, action in enumerate(all_actions):
                #         if env._match._action_parser._lookup_table[action[0]][5] == 1:
                #             someone_has_jumped = True
                #             jumpers.append(str(i))
                #     if someone_has_jumped:
                #         first_jumper_file.write(
                #             f"\n{str(progress.n)},{','.join(jumpers)}"
                #         )

            # get action indices, actions, and log probs for pretrained agents
            for idx in pretrained_idxs:
                policy = policies[idx]
                actions = policy.act(last_state, idx)
                # make sure output is in correct format
                if not isinstance(observations, np.ndarray):
                    actions = np.array(actions)

                # TODO: add converter that takes normal 8 actions into action space
                # actions = env._match._action_parser.convert_to_action_space(actions)
                all_actions[idx] = actions

            # to allow different action spaces, pad out short ones to longest length (assume later unpadding in parser)
            length = max([a.shape[0] for a in all_actions])
            padded_actions = np.array(
                [
                    np.pad(
                        a.astype("float64"),
                        (0, length - a.size),
                        "constant",
                        constant_values=np.NAN,
                    )
                    for a in all_actions
                ]
            )

            all_actions = padded_actions
            # TEST OUT ABOVE TO DEAL WITH VARIABLE LENGTH

            all_actions = np.vstack(all_actions)
            old_obs = observations
            observations, rewards, done, info = env.step(all_actions)
            truncated = False
            for terminal in env._match._terminal_conditions:  # noqa
                if isinstance(terminal, TruncatedCondition):
                    truncated |= terminal.is_truncated(info["state"])
            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]

            # prune data that belongs to old agents
            old_obs = [
                a for i, a in enumerate(old_obs) if latest_policy_indices[i] == 1
            ]
            all_indices = [
                d for i, d in enumerate(all_indices) if latest_policy_indices[i] == 1
            ]
            rewards = [
                r for i, r in enumerate(rewards) if latest_policy_indices[i] == 1
            ]
            all_log_probs = [
                r for i, r in enumerate(all_log_probs) if latest_policy_indices[i] == 1
            ]

            assert len(old_obs) == len(all_indices), (
                str(len(old_obs)) + " obs, " + str(len(all_indices)) + " ind"
            )
            assert len(old_obs) == len(rewards), (
                str(len(old_obs)) + " obs, " + str(len(rewards)) + " ind"
            )
            assert len(old_obs) == len(all_log_probs), (
                str(len(old_obs)) + " obs, " + str(len(all_log_probs)) + " ind"
            )
            assert len(old_obs) == len(rollouts), (
                str(len(old_obs)) + " obs, " + str(len(rollouts)) + " ind"
            )

            # Might be different if only one agent?
            if (
                not evaluate
            ):  # Evaluation matches can be long, no reason to keep them in memory
                trajectory_states.append(info["state"])
                for exp_buf, obs, act, rew, log_prob in zip(
                    rollouts, old_obs, all_indices, rewards, all_log_probs
                ):
                    exp_buf.add_step(
                        obs, act, rew, done + 2 * truncated, log_prob, info
                    )

            if progress is not None:
                progress.update()
                igt = progress.n * 8 / 120  # noqa
                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                if evaluate:
                    prog_str += f", BLUE {b} - {o} ORANGE"
                progress.set_postfix_str(prog_str)

            if done or truncated:
                result += info["result"]
                if info["result"] > 0:
                    b += 1
                elif info["result"] < 0:
                    o += 1

                if not evaluate:
                    break
                elif game_condition.done:  # noqa
                    break
                else:
                    observations, info = env.reset(return_info=True)

            last_state = info["state"]
            # v.visualizer.update()

    if scoreboard is not None:
        scoreboard.random_resets = random_resets  # noqa Checked above

    if progress is not None:
        progress.close()

    if evaluate:
        if isinstance(env._match._state_setter, DynamicGMSetter):  # noqa
            env._match._state_setter.setter = state_setter  # noqa
        else:
            env._match._state_setter = state_setter  # noqa
        env._match._terminal_conditions = terminals  # noqa
        env._match._reward_fn = reward  # noqa
        return result

    return rollouts, result, trajectory_states
