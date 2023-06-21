from typing import List

import numpy as np
import torch
from rlgym.gym import Gym
from rlgym_sim.utils.reward_functions.common_rewards import ConstantReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition
from tqdm import tqdm

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter


def generate_episode(env: Gym, policies, versions, eval_setter=DefaultState(), evaluate=False, scoreboard=None,
                     progress=False, selector_skip_k=None,
                     force_selector_choice=None,
                     unlock_selector_indices=None,
                     unlock_indices_group=None,
                     parser_boost_split=None,
                     selector_boost_skip_k=None,
                     initial_choice_block_indices=None,
                     initial_choice_block_weight=None,
                     ) -> (List[ExperienceBuffer], int):  # type: ignore
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
        game_condition = GameCondition(seconds_per_goal_forfeit=10 * 3,  # noqa
                                       max_overtime_seconds=300,
                                       max_no_touch_seconds=30)  # noqa
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
    result = 0

    last_state = info['state']  # game_state for obs_building of other agents
    pretrained_idx_tss_dict = {}
    for idx, policy in enumerate(policies):
        if isinstance(policy, HardcodedAgent):
            if hasattr(policy, "tick_skip_skip"):
                pretrained_idx_tss_dict[idx] = [policy.tick_skip_skip, 0]
            else:
                pretrained_idx_tss_dict[idx] = [1, 0]

    distinct_non_pretrained_versions_set = set([v for idx, v in enumerate(
        versions) if not isinstance(policies[idx], HardcodedAgent)])
    policy_version_idx_dict = {}
    for version in distinct_non_pretrained_versions_set:
        policy_version_idx_dict[version] = [
            idx for idx, v in enumerate(versions) if v == version]
    pretrained_idxs = [idx for idx, v in enumerate(
        versions) if isinstance(policies[idx], HardcodedAgent)]

    latest_policy_indices = [0 if isinstance(
        p, HardcodedAgent) else 1 for p in policies]
    # rollouts for all latest_policies
    rollouts = [
        ExperienceBuffer(infos=[info])
        for _ in range(sum(latest_policy_indices))
    ]

    b = o = 0
    with torch.no_grad():
        tick = [0] * len(policies)
        boost_tick = [0] * len(policies)
        do_selector = [True] * len(policies)
        do_boost = [True] * len(policies)
        last_actions = [None] * len(policies)
        last_boost = [None] * len(policies)
        first_step = True
        while True:
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
                actual_action_indices = [None] * len(idxs)
                if isinstance(observations[idxs[0]], tuple):
                    obs = tuple(np.concatenate([obs[i] for idx, obs in enumerate(observations) if idx in idxs], axis=0)
                                for i in range(len(observations[idxs[0]])))
                else:
                    obs = np.concatenate([obs for idx, obs in enumerate(
                        observations) if idx in idxs], axis=0)
                if initial_choice_block_weight is not None and first_step and np.random.random() <= initial_choice_block_weight:
                    dist = policy.get_action_distribution(obs, initial_choice_block_indices)
                else:
                    dist = policy.get_action_distribution(obs)
                action_indices = policy.sample_action(dist)
                action_indices_list = list(action_indices.numpy())
                if selector_skip_k is not None:
                    boost_list = [item[0] >= parser_boost_split for item in action_indices_list]
                # boost_list = [column[1] for column in action_indices_list]
                for i, idx in enumerate(idxs):
                    all_indices[idx] = action_indices_list[i]
                    actions = policy.env_compatible(action_indices[i])
                    if do_selector[idx]:
                        last_actions[idx] = actions
                        last_boost[idx] = boost_list[i]
                    elif selector_skip_k is not None and unlock_indices_group is not None and \
                            actions[0] in unlock_indices_group and last_actions[idx][0] in unlock_indices_group:
                        last_actions[idx] = actions
                        last_boost[idx] = boost_list[i]
                    else:
                        actions = last_actions[idx]
                        if selector_skip_k is not None and do_boost[idx]:
                            if boost_list[i] and actions[0] < parser_boost_split:
                                actions += parser_boost_split
                            elif not boost_list[i] and actions[0] >= parser_boost_split:
                                actions -= parser_boost_split
                            last_boost[idx] = boost_list[i]
                    # actions[1] = boost_list[i]
                    all_actions[idx] = actions
                    all_indices[idx] = actions
                    actual_action_indices[i] = actions
                action_indices = torch.tensor(np.array(actual_action_indices), dtype=torch.float64)
                log_probs = policy.log_prob(dist, action_indices)
                log_probs_list = list(log_probs.numpy())
                for i, idx in enumerate(idxs):
                    all_log_probs[idx] = log_probs_list[i]

            # get action indices, actions, and log probs for pretrained agents
            for idx in pretrained_idxs:
                policy = policies[idx]
                if pretrained_idx_tss_dict[idx][1] == 0:
                    actions = policy.act(last_state, idx)
                    # make sure output is in correct format
                    if not isinstance(observations, np.ndarray):
                        actions = np.array(actions)

                    # TODO: add converter that takes normal 8 actions into action space
                    # actions = env._match._action_parser.convert_to_action_space(actions)
                    all_actions[idx] = actions
                    last_actions[idx] = actions
                else:
                    all_actions[idx] = last_actions[idx]
                pretrained_idx_tss_dict[idx][1] = (
                                                          pretrained_idx_tss_dict[idx][1] + 1) % \
                                                  pretrained_idx_tss_dict[idx][0]

            # to allow different action spaces, pad out short ones to longest length (assume later unpadding in parser)
            length = max([a.shape[0] for a in all_actions])
            padded_actions = []
            for a in all_actions:
                action = np.pad(
                    a.astype('float64'), (0, length - a.size), 'constant', constant_values=np.NAN)
                padded_actions.append(action)

            all_actions = padded_actions
            # TEST OUT ABOVE TO DEAL WITH VARIABLE LENGTH

            all_actions = np.vstack(all_actions)
            old_obs = observations
            observations, rewards, done, info = env.step(all_actions)
            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]

            # need to check the force and stuff AFTER the parser has a look at the new action choice
            if selector_skip_k is not None:
                for i in range(len(do_selector)):
                    if not isinstance(policies[i], HardcodedAgent):
                        do_boost[i] = do_selector_action(selector_boost_skip_k, boost_tick[i])
                        do_selector[i] = do_selector_action(
                            selector_skip_k, tick[i])
                        if policies[i].deterministic or force_selector_choice[i]:
                            do_selector[i] = True
                            force_selector_choice[i] = False
                        if unlock_selector_indices is not None and all_actions[i][0] in unlock_selector_indices:
                            do_selector[i] = True
                    if do_selector[i]:
                        do_boost[i] = True
            else:
                do_selector = [True] * 6
                do_boost = [True] * 6
            for i in range(len(tick)):
                tick[i] = 0 if do_selector[i] else tick[i] + 1
                boost_tick[i] = 0 if do_boost[i] else boost_tick[i] + 1

            # prune data that belongs to old agents
            old_obs = [a for i, a in enumerate(
                old_obs) if latest_policy_indices[i] == 1]
            all_indices = [d for i, d in enumerate(
                all_indices) if latest_policy_indices[i] == 1]
            rewards = [r for i, r in enumerate(
                rewards) if latest_policy_indices[i] == 1]
            all_log_probs = [r for i, r in enumerate(
                all_log_probs) if latest_policy_indices[i] == 1]

            assert len(old_obs) == len(all_indices), str(
                len(old_obs)) + " obs, " + str(len(all_indices)) + " ind"
            assert len(old_obs) == len(rewards), str(
                len(old_obs)) + " obs, " + str(len(rewards)) + " ind"
            assert len(old_obs) == len(all_log_probs), str(
                len(old_obs)) + " obs, " + str(len(all_log_probs)) + " ind"
            assert len(old_obs) == len(rollouts), str(
                len(old_obs)) + " obs, " + str(len(rollouts)) + " ind"

            # Might be different if only one agent?
            if not evaluate:  # Evaluation matches can be long, no reason to keep them in memory
                for exp_buf, obs, act, rew, log_prob in zip(rollouts, old_obs, all_indices, rewards, all_log_probs):
                    exp_buf.add_step(obs, act, rew, done, log_prob, info)

            if progress is not None:
                progress.update()
                igt = progress.n * env._match._tick_skip / 120  # noqa
                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                if evaluate:
                    prog_str += f", BLUE {b} - {o} ORANGE"
                progress.set_postfix_str(prog_str)

            if done:
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

            last_state = info['state']

            first_step = False

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

    return rollouts, result


def do_selector_action(selector_skip_k, tick) -> bool:
    p = 1 / (1 + (selector_skip_k * tick))
    if np.random.uniform() < p:
        return False
    else:
        return True
