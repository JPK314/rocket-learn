import copy
import functools
import io
import itertools
import os
import sqlite3 as sql
import time
from threading import Thread
from uuid import uuid4

import numpy as np
from redis import Redis
from rlgym.envs import Match
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym
from rlgym.utils.state_setters import DefaultState
from rocketsimvisualizer import VisualizerThread
from tabulate import tabulate
from termcolor import colored

import rocket_learn.agent.policy
import rocket_learn.utils.generate_episode
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.agent.types import PretrainedAgents
from rocket_learn.matchmaker.base_matchmaker import BaseMatchmaker
from rocket_learn.rollout_generator.redis.utils import (
    EXPERIENCE_PER_MODE,
    LATEST_RATING_ID,
    MODEL_LATEST,
    OPPONENT_MODELS,
    ROLLOUTS,
    VERSION_LATEST,
    WORKER_IDS,
    _serialize,
    _unserialize_model,
    decode_buffers,
    encode_buffers,
    get_rating,
    get_ratings,
)
from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter
from rocket_learn.utils.util import probability_NvsM


class RedisRolloutWorker:
    """
    Provides RedisRolloutGenerator with rollouts via a Redis server

     :param redis: redis object
     :param name: rollout worker name
     :param match: match object
     :param matchmaker: BaseMatchmaker object
     :param evaluation_prob: Odds of running an evaluation match
     :param dynamic_gm: Pick game mode dynamically. If True, Match.team_size should be 3
     :param streamer_mode: Should run in streamer mode (less data printed to screen)
     :param send_gamestates: Should gamestate data be sent back (increases data sent) - must send obs or gamestates
     :param send_obs: Should observations be send back (increases data sent) - must send obs or gamestates
     :param scoreboard: Scoreboard object
     :param pretrained_agents: PretrainedAgents typed dict
     :param human_agent: human agent object. Sets a human match if not None
     :param force_paging: Should paging be forced
     :param auto_minimize: automatically minimize the launched rocket league instance
     :param local_cache_name: name of local database used for model caching. If None, caching is not used
     :param gamemode_weights: dict of dynamic gamemode choice weights. If None, default equal experience
     :param gamemode_weight_ema_alpha: alpha for the exponential moving average of gamemode weighting
     :param selector_skip_k: value to control the tick skip probability of the selector
    """

    def __init__(
        self,
        redis: Redis,
        name: str,
        match: Match,
        matchmaker: BaseMatchmaker,
        evaluation_prob=0.01,
        dynamic_gm=True,
        streamer_mode=False,
        send_gamestates=True,
        send_obs=True,
        scoreboard=None,
        pretrained_agents: PretrainedAgents = None,
        human_agent=None,
        force_paging=False,
        auto_minimize=True,
        tick_skip=0,
        local_cache_name=None,
        force_old_deterministic=False,
        force_current_deterministic=False,
        deterministic_streamer=False,
        gamemode_weights=None,
        batch_mode=False,
        step_size=100_000,
        pipeline_limit_bytes=10_000_000,
        gamemode_weight_ema_alpha=0.02,
        selector_skip_k=None,
        eval_setter=DefaultState(),
        epic_rl_exe_path=None,
        simulator=False,
        visualize=False,
        dodge_deadzone=0.5,
        live_progress=True,
    ):
        # TODO model or config+params so workers can recreate just from redis connection?
        self.eval_setter = eval_setter
        self.redis = redis
        self.name = name

        self.matchmaker = matchmaker

        self.selector_skip_k = selector_skip_k
        self.force_selector_choice = [False] * 6

        assert send_gamestates or send_obs, "Must have at least one of obs or states"

        self.pretrained_agents = {}
        self.pretrained_agents_keymap = {}
        if pretrained_agents is not None:
            self.pretrained_agents = pretrained_agents
            for agent, config in pretrained_agents.items():
                self.pretrained_agents_keymap[config["key"]] = agent

        self.human_agent = human_agent
        self.force_old_deterministic = force_old_deterministic
        self.force_current_deterministic = force_current_deterministic

        if human_agent and pretrained_agents:
            print("** WARNING - Human Player and Pretrained Agents are in conflict. **")
            print("**           Pretrained Agents will be ignored.                  **")

        self.streamer_mode = streamer_mode
        self.deterministic_streamer = deterministic_streamer

        self.current_agent = _unserialize_model(self.redis.get(MODEL_LATEST))
        if self.streamer_mode and self.deterministic_streamer:
            self.current_agent.deterministic = True
        self.evaluation_prob = evaluation_prob
        self.send_gamestates = send_gamestates
        self.send_obs = send_obs
        self.dynamic_gm = dynamic_gm
        self.gamemode_weights = gamemode_weights
        if self.gamemode_weights is None:
            self.gamemode_weights = {"1v1": 1 / 3, "2v2": 1 / 3, "3v3": 1 / 3}
        assert np.isclose(
            sum(self.gamemode_weights.values()), 1
        ), "gamemode_weights must sum to 1"
        self.target_weights = copy.copy(self.gamemode_weights)
        # change weights from percentage of experience desired to percentage of gamemodes necessary (approx)
        self.current_weights = copy.copy(self.gamemode_weights)
        for k in self.current_weights.keys():
            b, o = k.split("v")
            self.current_weights[k] /= int(b)
        self.current_weights = {
            k: self.current_weights[k] / (sum(self.current_weights.values()) + 1e-8)
            for k in self.current_weights.keys()
        }
        self.mean_exp_grant = {}
        for k in self.gamemode_weights.keys():
            b, o = k.split("v")
            size = int(b) + int(o)
            self.mean_exp_grant[k] = 500 * size
        self.ema_alpha = gamemode_weight_ema_alpha
        self.local_cache_name = local_cache_name

        self.uuid = str(uuid4())
        self.redis.rpush(WORKER_IDS, self.uuid)

        self.batch_mode = batch_mode
        self.step_size_limit = min(step_size / 20, 25_000)
        if self.batch_mode:
            self.red_pipe = self.redis.pipeline()
            self.step_last_send = 0
        self.pipeline_size = 0
        self.pipeline_limit = pipeline_limit_bytes  # 10 MB is default

        # currently doesn't rebuild, if the old is there, reuse it.
        if self.local_cache_name:
            self.sql = sql.connect("redis-model-cache-" + local_cache_name + ".db")
            # if the table doesn't exist in the database, make it
            self.sql.execute(
                """
                CREATE TABLE if not exists MODELS (
                    id TEXT PRIMARY KEY,
                    parameters BLOB NOT NULL
                );
            """
            )

        if not self.streamer_mode:
            print(
                "Started worker",
                self.uuid,
                "on host",
                self.redis.connection_pool.connection_kwargs.get("host"),
                "under name",
                name,
            )  # TODO log instead
        else:
            print("Streaming mode set. Running silent.")

        self.scoreboard = scoreboard
        state_setter = DynamicGMSetter(
            match._state_setter
        )  # noqa Rangler made me do it
        self.set_team_size = state_setter.set_team_size
        match._state_setter = state_setter
        self.match = match
        if simulator:
            import rlgym_sim

            self.env = rlgym_sim.gym.Gym(
                match=self.match,
                copy_gamestate_every_step=True,
                dodge_deadzone=dodge_deadzone,
                tick_skip=tick_skip,
                gravity=1,
                boost_consumption=1,
            )
        else:
            self.env = Gym(
                match=self.match,
                pipe_id=os.getpid(),
                launch_preference=LaunchPreference.EPIC,
                use_injector=True,
                force_paging=force_paging,
                raise_on_crash=True,
                auto_minimize=auto_minimize,
            )
        self.total_steps_generated = 0
        self.live_progress = live_progress

    @functools.lru_cache(maxsize=8)
    def _get_past_model(self, version):
        # if version in local database, query from database
        # if not, pull from REDIS and store in disk cache

        if self.local_cache_name:
            models = self.sql.execute(
                "SELECT parameters FROM MODELS WHERE id == ?", (version,)
            ).fetchall()
            if len(models) == 0:
                bytestream = self.redis.hget(OPPONENT_MODELS, version)
                model = _unserialize_model(bytestream)

                self.sql.execute(
                    "INSERT INTO MODELS (id, parameters) VALUES (?, ?)",
                    (version, bytestream),
                )
                self.sql.commit()
            else:
                # should only ever be 1 version of parameters
                assert len(models) <= 1
                # stored as tuple due to sqlite,
                assert len(models[0]) == 1

                bytestream = models[0][0]
                model = _unserialize_model(bytestream)
        else:
            model = _unserialize_model(self.redis.hget(OPPONENT_MODELS, version))

        return model

    def select_gamemode(self, equal_likelihood):

        emp_weight = {
            k: self.mean_exp_grant[k] / (sum(self.mean_exp_grant.values()) + 1e-8)
            for k in self.mean_exp_grant.keys()
        }
        cor_weight = {
            k: self.gamemode_weights[k] / emp_weight[k]
            for k in self.gamemode_weights.keys()
        }
        self.current_weights = {
            k: cor_weight[k] / (sum(cor_weight.values()) + 1e-8) for k in cor_weight
        }
        mode = np.random.choice(
            list(self.current_weights.keys()), p=list(self.current_weights.values())
        )
        if equal_likelihood:
            mode = np.random.choice(list(self.current_weights.keys()))
        b, o = mode.split("v")
        return int(b), int(o)

    @staticmethod
    def make_table(versions, ratings, blue, orange, streamer=False):
        version_info = []
        for v, r in zip(versions, ratings):
            if v == "na":
                version_info.append(["Human", "N/A"])
            else:
                if isinstance(v, int):
                    v *= -1
                version_info.append([v, f"{r.mu:.2f}±{2 * r.sigma:.2f}"])

        blue_versions, blue_ratings = list(zip(*version_info[:blue]))
        orange_versions, orange_ratings = (
            list(zip(*version_info[blue:])) if orange > 0 else list(((0,), ("N/A",)))
        )
        orange_versions2 = [str(v) for v in orange_versions]
        blue_versions2 = [str(v) for v in blue_versions]

        blue_versions2 = ["LATEST" if v.isdigit() else v for v in blue_versions2]
        orange_versions2 = ["LATEST" if v.isdigit() else v for v in orange_versions2]
        if streamer:
            with open("Blue.txt", "w") as file:
                file.write("  ".join(blue_versions2))
                file.write("\n")

            with open("Orange.txt", "w") as file:
                file.write("  ".join(orange_versions2))
                file.write("\n")

        if blue < orange:
            blue_versions += ("",) * (orange - blue)
            blue_ratings += ("",) * (orange - blue)
        elif orange < blue:
            orange_versions += ("",) * (blue - orange)
            orange_ratings += ("",) * (blue - orange)

        table_str = tabulate(
            list(zip(blue_versions, blue_ratings, orange_versions, orange_ratings)),
            headers=["Blue", "rating", "Orange", "rating"],
            tablefmt="rounded_outline",
        )

        return table_str

    def run(self, v: VisualizerThread = None):  # Mimics Thread
        """
        begin processing in already launched match and push to redis
        """
        n = 0
        latest_version = None
        # t = Thread()
        # t.start()
        while True:
            # Get the most recent version available
            available_version = self.redis.get(VERSION_LATEST)
            if available_version is None:
                time.sleep(1)
                continue  # Wait for version to be published (not sure if this is necessary?)
            available_version = int(available_version)

            # Only try to download latest version when new
            if latest_version != available_version:
                model_bytes = self.redis.get(MODEL_LATEST)
                if model_bytes is None:
                    time.sleep(1)
                    continue  # This is maybe not necessary? Can't hurt to leave it in.
                latest_version = available_version
                updated_agent = _unserialize_model(model_bytes)
                self.current_agent = updated_agent
                if self.force_current_deterministic or (
                    self.streamer_mode and self.deterministic_streamer
                ):
                    self.current_agent.deterministic = True

            n += 1

            evaluate = np.random.random() < self.evaluation_prob

            if self.dynamic_gm:
                blue, orange = self.select_gamemode(equal_likelihood=evaluate)
            elif self.match.agents == 1:
                blue = 1
                orange = 0
            elif self.match._spawn_opponents is False:  # noqa
                blue = self.match.agents
                orange = 0
            else:
                blue = orange = self.match.agents // 2

            if self.human_agent:
                n_new = blue + orange - 1
                versions = ["human"]

                agents = [self.human_agent]
                for n in range(n_new):
                    agents.append(self.current_agent)
                    versions.append(latest_version)

                versions = [v if v != -1 else latest_version for v in versions]
                ratings = ["na"] * len(versions)
            else:
                (
                    versions,
                    ratings,
                    evaluate,
                    blue,
                    orange,
                ) = self.matchmaker.generate_matchup(
                    self.redis, blue + orange, evaluate
                )
                agents = []
                latest_version_idxs = []
                non_pretrained_idxs = []
                for i, version in enumerate(versions):
                    if version == -1:
                        versions[i] = latest_version
                        agents.append(self.current_agent)
                        latest_version_idxs.append(i)
                        non_pretrained_idxs.append(i)
                    else:
                        # For instances of PretrainedDiscretePolicy, whose redis qualities keys end with -deterministic or -stochastic
                        short_name = "-".join(version.split("-")[:-1])
                        if short_name in self.pretrained_agents_keymap:
                            selected_agent = self.pretrained_agents_keymap[short_name]
                        # For any other instances of HardcodedAgent, whose redis qualities keys are just the key in the keymap
                        elif version in self.pretrained_agents_keymap:
                            selected_agent = self.pretrained_agents_keymap[version]
                        else:
                            selected_agent = self._get_past_model(short_name)
                            if self.force_old_deterministic and not evaluate:
                                versions[i] = versions[i].replace(
                                    "stochastic", "deterministic"
                                )
                                version = version.replace("stochastic", "deterministic")
                            non_pretrained_idxs.append(i)

                        if isinstance(selected_agent, DiscretePolicy):
                            if version.endswith("deterministic"):
                                selected_agent.deterministic = True
                            elif version.endswith("stochastic"):
                                selected_agent.deterministic = False
                            else:
                                raise ValueError("Unknown version type")
                        agents.append(selected_agent)

            self.set_team_size(blue, orange)

            if not self.streamer_mode:

                table_str = self.make_table(
                    versions, ratings, blue, orange, streamer=False
                )
            else:
                table_str = self.make_table(
                    versions, ratings, blue, orange, streamer=True
                )

            if evaluate and not self.streamer_mode and self.human_agent is None:
                print(colored("EVALUATION GAME", "magenta") + table_str)
                result = rocket_learn.utils.generate_episode.generate_episode(
                    self.env,
                    agents,
                    versions,
                    evaluate=True,
                    scoreboard=self.scoreboard,
                    progress=self.live_progress,
                    eval_setter=self.eval_setter,
                    v=v,
                )
                rollouts = []
                print(
                    colored("Evaluation finished, goal differential:", "magenta"),
                    result,
                )
                print()
            else:
                if not self.streamer_mode:
                    print(colored("ROLLOUT\n", "magenta") + table_str)

                try:
                    (
                        rollouts,
                        result,
                        trajectory_states,
                    ) = rocket_learn.utils.generate_episode.generate_episode(
                        self.env,
                        agents,
                        versions,
                        evaluate=False,
                        scoreboard=self.scoreboard,
                        v=v,
                    )

                    # Happens sometimes, unknown reason
                    if len(rollouts[0].observations) <= 1:
                        print(" ** Rollout Generation Error: Restarting Generation ** ")
                        print()
                        continue
                except EnvironmentError:
                    self.env.attempt_recovery()
                    continue

                state = rollouts[0].infos[-2]["state"]
                goal_speed = np.linalg.norm(state.ball.linear_velocity) * 0.036  # kph
                str_result = ("+" if result > 0 else "") + str(result)
                episode_exp = len(rollouts[0].observations) * len(rollouts)
                self.total_steps_generated += episode_exp
                if self.dynamic_gm:
                    old_exp = self.mean_exp_grant[f"{blue}v{orange}"]
                    self.mean_exp_grant[f"{blue}v{orange}"] = (
                        (episode_exp - old_exp) * self.ema_alpha
                    ) + old_exp
                post_stats = f"Rollout finished after {colored(len(rollouts[0].observations), 'magenta')} steps ({colored(self.total_steps_generated, 'magenta')} total steps), result was {colored(str_result, 'magenta')}"
                if result != 0:
                    post_stats += f", goal speed: {goal_speed:.2f} kph\nScoreline is {int(state.blue_score)} - {int(state.orange_score)}\nDifferential is {int(state.blue_score)- int(state.orange_score)}"
                # print(post_stats)

                if not self.streamer_mode:
                    print(post_stats)
                    print()

            if not self.streamer_mode and not evaluate:
                # Get aux head labels here. Only need to do for latest versions
                latest_version_car_ids = [
                    state.players[idx].car_id for idx in latest_version_idxs
                ]
                latest_version_buffer_idxs = [
                    i
                    for i, idx in enumerate(non_pretrained_idxs)
                    if idx in latest_version_idxs
                ]
                latest_rewards = [
                    rollouts[i].rewards
                    for i, idx in enumerate(non_pretrained_idxs)
                    if idx in latest_version_buffer_idxs
                ]
                latest_observations = [
                    rollouts[i].observations
                    for i, idx in enumerate(non_pretrained_idxs)
                    if idx in latest_version_buffer_idxs
                ]
                aux_labels = self.current_agent.get_aux_heads_labels(
                    latest_rewards,
                    trajectory_states,
                    latest_version_car_ids,
                    latest_observations,
                )
                all_aux_labels = [None] * len(versions)
                aux_label_idx = 0
                for idx in range(len(versions)):
                    if idx in latest_version_idxs:
                        all_aux_labels[idx] = tuple(
                            aux_head_labels[aux_label_idx]
                            for aux_head_labels in aux_labels
                        )
                        aux_label_idx += 1

            if evaluate:
                all_aux_labels = None

            if not self.streamer_mode and not self.batch_mode:

                rollout_data = encode_buffers(
                    rollouts,
                    all_aux_labels,
                    return_obs=self.send_obs,
                    return_states=self.send_gamestates,
                    return_rewards=True,
                )
                # sanity_check = decode_buffers(rollout_data, versions,
                #                               has_obs=False, has_states=True, has_rewards=True,
                #                               obs_build_factory=lambda: self.match._obs_builder,
                #                               rew_func_factory=lambda: self.match._reward_fn,
                #                               act_parse_factory=lambda: self.match._action_parser)
                rollout_bytes = _serialize(
                    (
                        rollout_data,
                        versions,
                        self.uuid,
                        self.name,
                        result,
                        self.send_obs,
                        self.send_gamestates,
                        True,
                    )
                )

                # while True:
                # t.join()

                def send():
                    n_items = self.redis.rpush(ROLLOUTS, rollout_bytes)
                    if n_items >= 1000:
                        print(
                            "Had to limit rollouts. Learner may have have crashed, or is overloaded"
                        )
                        self.redis.ltrim(ROLLOUTS, -100, -1)

                send()
                # t = Thread(target=send)
                # t.start()
                # time.sleep(0.01)

            elif not self.streamer_mode and self.batch_mode:

                rollout_data = encode_buffers(
                    rollouts,
                    all_aux_labels,
                    return_obs=self.send_obs,
                    return_states=self.send_gamestates,
                    return_rewards=True,
                )
                rollout_bytes = _serialize(
                    (
                        rollout_data,
                        versions,
                        self.uuid,
                        self.name,
                        result,
                        self.send_obs,
                        self.send_gamestates,
                        True,
                    )
                )

                self.pipeline_size += len(rollout_bytes)

                self.red_pipe.rpush(ROLLOUTS, rollout_bytes)

                #  def send():
                if (
                    (self.total_steps_generated - self.step_last_send)
                    > self.step_size_limit
                    or len(self.red_pipe) > 100
                    or self.pipeline_size > self.pipeline_limit
                ):
                    n_items = self.red_pipe.execute()
                    self.pipeline_size = 0
                    if n_items[-1] >= 10000:
                        print(
                            "Had to limit rollouts. Learner may have have crashed, or is overloaded"
                        )
                        self.redis.ltrim(ROLLOUTS, -100, -1)
                    self.step_last_send = self.total_steps_generated
