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


def generate_episode(env: Gym, policies, versions, eval_setter=DefaultState(), evaluate=False, scoreboard=None, progress=True, silent_mode=False, v: VisualizerThread = None) -> (List[ExperienceBuffer], int):  # type: ignore
    """
    create experience buffer data by interacting with the environment(s)
    """
    if progress and not silent_mode:
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
    b = o = 0
    with torch.no_grad():
        while True:
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
                body_out = policy(obs)
                dist = policy.get_action_distribution(body_out)
                action_indices = policy.sample_action(dist)
                log_probs = policy.log_prob(dist, action_indices)
                action_indices_list = list(action_indices.numpy())
                log_probs_list = list(log_probs.numpy())
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
