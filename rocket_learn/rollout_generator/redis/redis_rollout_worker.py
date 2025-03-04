import functools
import itertools
import os
import time
from threading import Thread
from uuid import uuid4

import sqlite3 as sql

import numpy as np
from redis import Redis
from rlgym.envs import Match
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym

import rocket_learn.agent.policy
import rocket_learn.utils.generate_episode
from rocket_learn.rollout_generator.redis.utils import _unserialize_model, MODEL_LATEST, WORKER_IDS, OPPONENT_MODELS, \
    VERSION_LATEST, _serialize, ROLLOUTS, encode_buffers, decode_buffers, get_rating, LATEST_RATING_ID, \
    EXPERIENCE_PER_MODE
from rocket_learn.utils.util import probability_NvsM
from rocket_learn.utils.dynamic_gamemode_setter import DynamicGMSetter


class RedisRolloutWorker:
    """
    Provides RedisRolloutGenerator with rollouts via a Redis server

     :param redis: redis object
     :param name: rollout worker name
     :param match: match object
     :param past_version_prob: Odds of playing against previous checkpoints
     :param evaluation_prob: Odds of running an evaluation match
     :param sigma_target: Trueskill sigma target
     :param dynamic_gm: Pick game mode dynamically. If True, Match.team_size should be 3
     :param streamer_mode: Should run in streamer mode (less data printed to screen)
     :param send_gamestates: Should gamestate data be sent back (increases data sent) - must send obs or gamestates
     :param send_obs: Should observations be send back (increases data sent) - must send obs or gamestates
     :param scoreboard: Scoreboard object
     :param pretrained_agents: Dict{} of pretrained agents and their appearance probability
     :param human_agent: human agent object. Sets a human match if not None
     :param force_paging: Should paging be forced
     :param auto_minimize: automatically minimize the launched rocket league instance
     :param local_cache_name: name of local database used for model caching. If None, caching is not used
     :param gamemode_weights: dict of dynamic gamemode choice weights. If None, default equal experience
    """

    def __init__(self, redis: Redis, name: str, match: Match,
                 past_version_prob=.2, evaluation_prob=0.01, sigma_target=1,
                 dynamic_gm=True, streamer_mode=False, send_gamestates=True,
                 send_obs=True, scoreboard=None, pretrained_agents=None,
                 human_agent=None, force_paging=False, auto_minimize=True,
                 local_cache_name=None, gamemode_weights=None, full_team_evaluations=False):
        # TODO model or config+params so workers can recreate just from redis connection?
        self.redis = redis
        self.name = name

        assert send_gamestates or send_obs, "Must have at least one of obs or states"

        self.pretrained_agents = {}
        self.pretrained_total_prob = 0
        if pretrained_agents is not None:
            self.pretrained_agents = pretrained_agents
            self.pretrained_total_prob = sum([self.pretrained_agents[key] for key in self.pretrained_agents])

        self.human_agent = human_agent

        if human_agent and pretrained_agents:
            print("** WARNING - Human Player and Pretrained Agents are in conflict. **")
            print("**           Pretrained Agents will be ignored.                  **")

        self.streamer_mode = streamer_mode

        self.current_agent = _unserialize_model(self.redis.get(MODEL_LATEST))

        self.past_version_prob = past_version_prob
        self.evaluation_prob = evaluation_prob
        self.sigma_target = sigma_target
        self.send_gamestates = send_gamestates
        self.send_obs = send_obs
        self.dynamic_gm = dynamic_gm
        self.gamemode_weights = gamemode_weights
        if self.gamemode_weights is not None:
            assert np.isclose(sum(self.gamemode_weights.values()), 1), "gamemode_weights must sum to 1"
        self.gamemode_exp_per_episode_ema = {}
        self.local_cache_name = local_cache_name

        self.full_team_evaluations = full_team_evaluations

        self.uuid = str(uuid4())
        self.redis.rpush(WORKER_IDS, self.uuid)

        # currently doesn't rebuild, if the old is there, reuse it.
        if self.local_cache_name:
            self.sql = sql.connect('redis-model-cache-' + local_cache_name + '.db')
            # if the table doesn't exist in the database, make it
            self.sql.execute("""
                CREATE TABLE if not exists MODELS (
                    id TEXT PRIMARY KEY,
                    parameters BLOB NOT NULL
                );
            """)

        if not self.streamer_mode:
            print("Started worker", self.uuid, "on host", self.redis.connection_pool.connection_kwargs.get("host"),
                  "under name", name)  # TODO log instead
        else:
            print("Streaming mode set. Running silent.")

        self.scoreboard = scoreboard
        state_setter = DynamicGMSetter(match._state_setter)  # noqa Rangler made me do it
        self.set_team_size = state_setter.set_team_size
        match._state_setter = state_setter
        self.match = match
        self.env = Gym(match=self.match, pipe_id=os.getpid(), launch_preference=LaunchPreference.EPIC,
                       use_injector=True, force_paging=force_paging, raise_on_crash=True, auto_minimize=auto_minimize)
        self.total_steps_generated = 0

    def _get_opponent_ids(self, n_new, n_old, pretrained_choice):
        # Get qualities
        assert (n_new + n_old) % 2 == 0
        per_team = (n_new + n_old) // 2
        gamemode = f"{per_team}v{per_team}"
        latest_id = self.redis.get(LATEST_RATING_ID).decode("utf-8")
        latest_key = f"{latest_id}-stochastic"
        if n_old == 0:
            rating = get_rating(gamemode, latest_key, self.redis)
            return [-1] * n_new, [rating] * n_new

        ratings = get_rating(gamemode, None, self.redis)
        latest_rating = ratings[latest_key]
        keys, values = zip(*ratings.items())

        if n_new == 0 and len(values) >= n_old:  # Evaluation game, try to find agents with high sigma
            sigmas = np.array([r.sigma for r in values])
            probs = np.clip(sigmas - self.sigma_target, a_min=0, a_max=None)
            s = probs.sum()
            if s == 0:  # No versions with high sigma available
                if np.random.normal(0, self.sigma_target) > 1:
                    # Some chance of doing a match with random versions, so they might correct themselves
                    probs = np.ones_like(probs) / len(probs)
                else:
                    return [-1] * n_old, [latest_rating] * n_old
            else:
                probs /= s
            versions = [np.random.choice(len(keys), p=probs)]
            if self.full_team_evaluations:
                versions = versions * per_team
            target_rating = values[versions[0]]
            n_old -= 1
        elif pretrained_choice is not None:  # pretrained agent chosen, just need index generation
            matchups = np.full((n_new + n_old), -1).tolist()
            for i in range(n_old):
                index = np.random.randint(0, n_new + n_old)
                matchups[index] = 'na'
            return matchups, ratings.values()
        else:
            if n_new == 0:  # Would-be evaluation game, but not enough agents
                n_new = n_old
                n_old = 0
            versions = [-1] * n_new
            target_rating = latest_rating

        # Calculate 1v1 win prob against target
        # All the agents included should hold their own (at least approximately)
        # This is to prevent unrealistic scenarios,
        # like for instance ratings of [100, 0] vs [100, 0], which is technically fair but not useful
        probs = np.zeros(len(keys))
        if n_new == 0 and self.full_team_evaluations:
            for i, rating in enumerate(values):
                if i == versions[0]:
                    p = 0  # Don't add more of the same agent in evaluation matches
                else:
                    p = probability_NvsM([rating] * per_team, [target_rating] * per_team)
                probs[i] = p
            opponent = np.random.choice(len(probs), p=probs)
            if np.random.random() < 0.5:  # Randomly do blue/orange
                versions = versions + [opponent] * per_team
            else:
                versions = [opponent] * per_team + versions
            return [keys[i] for i in versions], [values[i] for i in versions]
        else:
            for i, rating in enumerate(values):
                if n_new == 0 and i == versions[0]:
                    continue  # Don't add more of the same agent in evaluation matches
                p = probability_NvsM([rating], [target_rating])
                probs[i] = (p * (1 - p)) ** ((n_new + n_old) // 2)  # Be less lenient the more players there are
            probs /= probs.sum()

            old_versions = np.random.choice(len(probs), size=n_old, p=probs, replace=True).tolist()
            versions += old_versions

            # Then calculate the full matchup, with just permutations of the selected versions (weighted by fairness)
            matchups = []
            qualities = []
            for perm in itertools.permutations(versions):
                it_ratings = [latest_rating if v == -1 else values[v] for v in perm]
                mid = len(it_ratings) // 2
                p = probability_NvsM(it_ratings[:mid], it_ratings[mid:])
                if n_new == 0 and set(perm[:mid]) == set(perm[mid:]):  # Don't want team against team
                    p = 0
                matchups.append(perm)
                qualities.append(p * (1 - p))  # From AlphaStar
            qualities = np.array(qualities)
            s = qualities.sum()
            if s == 0:
                return [-1] * (n_new + n_old), [latest_rating] * (n_new + n_old)
            k = np.random.choice(len(matchups), p=qualities / s)
            return [-1 if i == -1 else keys[i] for i in matchups[k]], \
                   [latest_rating if i == -1 else values[i] for i in matchups[k]]

    @functools.lru_cache(maxsize=8)
    def _get_past_model(self, version):
        # if version in local database, query from database
        # if not, pull from REDIS and store in disk cache

        if self.local_cache_name:
            models = self.sql.execute("SELECT parameters FROM MODELS WHERE id == ?", (version,)).fetchall()
            if len(models) == 0:
                bytestream = self.redis.hget(OPPONENT_MODELS, version)
                model = _unserialize_model(bytestream)

                self.sql.execute('INSERT INTO MODELS (id, parameters) VALUES (?, ?)', (version, bytestream))
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
        mode_exp = {m.decode("utf-8"): int(v) for m, v in self.redis.hgetall(EXPERIENCE_PER_MODE).items()}
        modes = list(mode_exp.keys())
        if equal_likelihood:
            mode = np.random.choice(modes)
        else:
            dist = np.array(list(mode_exp.values())) + 1
            dist = dist / dist.sum()
            if self.gamemode_weights is None:
                target_dist = np.ones(len(modes))
            else:
                target_dist = np.array([self.gamemode_weights[k] for k in modes])
            mode_steps_per_episode = np.array(list(self.gamemode_exp_per_episode_ema.get(m, None) or 1 for m in modes))

            target_dist = target_dist / mode_steps_per_episode
            target_dist = target_dist / target_dist.sum()
            inv_dist = 1 - dist
            inv_dist = inv_dist / inv_dist.sum()

            dist = target_dist * inv_dist
            dist = dist / dist.sum()

            mode = np.random.choice(modes, p=dist)

        b, o = mode.split("v")
        return int(b), int(o)

    def run(self):  # Mimics Thread
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

            n += 1
            pretrained_choice = None

            evaluate = np.random.random() < self.evaluation_prob

            if self.dynamic_gm:
                blue, orange = self.select_gamemode(equal_likelihood=evaluate or self.streamer_mode)
            elif self.match._spawn_opponents is False:
                blue = self.match.agents
                orange = 0
            else:
                blue = orange = self.match.agents // 2
            self.set_team_size(blue, orange)

            if self.human_agent:
                n_new = blue + orange - 1
                versions = ['na']

                agents = [self.human_agent]
                for n in range(n_new):
                    agents.append(self.current_agent)
                    versions.append(-1)

                versions = [v if v != -1 else latest_version for v in versions]
                ratings = ["na"] * len(versions)
            else:
                # TODO customizable past agent selection, should team only be same agent?
                agents, pretrained_choice, versions, ratings = self._generate_matchup(blue + orange,
                                                                                      latest_version,
                                                                                      pretrained_choice,
                                                                                      evaluate)

            evaluate = not any(isinstance(v, int) and v < 0 for v in versions)  # Might be changed in matchup code

            version_info = []
            for v, r in zip(versions, ratings):
                if pretrained_choice is not None and v == 'na':  # print name but don't send it back
                    version_info.append(str(type(pretrained_choice).__name__))
                elif v == 'na':
                    version_info.append('Human_player')
                else:
                    version_info.append(f"{v} ({r.mu:.2f}±{2 * r.sigma:.2f})")

            if evaluate and not self.streamer_mode and self.human_agent is None:
                print("Running evaluation game with versions:", version_info)
                result = rocket_learn.utils.generate_episode.generate_episode(self.env, agents, evaluate=True,
                                                                              scoreboard=self.scoreboard)
                rollouts = []
                print("Evaluation finished, goal differential:", result)
            else:
                if not self.streamer_mode:
                    print("Generating rollout with versions:", version_info)

                try:
                    rollouts, result = rocket_learn.utils.generate_episode.generate_episode(self.env, agents,
                                                                                            evaluate=False,
                                                                                            scoreboard=self.scoreboard)

                    if len(rollouts[0].observations) <= 1:  # Happens sometimes, unknown reason
                        print(" ** Rollout Generation Error: Restarting Generation ** ")
                        continue
                except EnvironmentError:
                    self.env.attempt_recovery()
                    continue

                state = rollouts[0].infos[-2]["state"]
                goal_speed = np.linalg.norm(state.ball.linear_velocity) * 0.036  # kph
                str_result = ('+' if result > 0 else "") + str(result)
                episode_exp = len(rollouts[0].observations) * len(rollouts)
                self.total_steps_generated += episode_exp

                if self.dynamic_gm and not evaluate:
                    mode = f"{blue}v{orange}"
                    if mode in self.gamemode_exp_per_episode_ema:
                        current_mean = self.gamemode_exp_per_episode_ema[mode]
                        self.gamemode_exp_per_episode_ema[mode] = 0.98 * current_mean + 0.02 * episode_exp
                    else:
                        self.gamemode_exp_per_episode_ema[mode] = episode_exp

                post_stats = f"Rollout finished after {len(rollouts[0].observations)} steps ({self.total_steps_generated} total steps), result was {str_result}"
                if result != 0:
                    post_stats += f", goal speed: {goal_speed:.2f} kph"

                if not self.streamer_mode:
                    print(post_stats)

            if not self.streamer_mode:
                rollout_data = encode_buffers(rollouts,
                                              return_obs=self.send_obs,
                                              return_states=self.send_gamestates,
                                              return_rewards=True)
                # sanity_check = decode_buffers(rollout_data, versions,
                #                               has_obs=False, has_states=True, has_rewards=True,
                #                               obs_build_factory=lambda: self.match._obs_builder,
                #                               rew_func_factory=lambda: self.match._reward_fn,
                #                               act_parse_factory=lambda: self.match._action_parser)
                rollout_bytes = _serialize((rollout_data, versions, self.uuid, self.name, result,
                                            self.send_obs, self.send_gamestates, True))

                # while True:
                # t.join()

                def send():
                    n_items = self.redis.rpush(ROLLOUTS, rollout_bytes)
                    if n_items >= 1000:
                        print("Had to limit rollouts. Learner may have have crashed, or is overloaded")
                        self.redis.ltrim(ROLLOUTS, -100, -1)

                send()
                # t = Thread(target=send)
                # t.start()
                # time.sleep(0.01)

    def _generate_matchup(self, n_agents, latest_version, pretrained_choice, evaluate):
        if evaluate:
            n_old = n_agents
        else:
            n_old = 0
            rand_choice = np.random.random()
            if rand_choice < self.past_version_prob:
                n_old = np.random.randint(low=1, high=n_agents)
            elif rand_choice < (self.past_version_prob + self.pretrained_total_prob):
                wheel_prob = self.past_version_prob
                for agent in self.pretrained_agents:
                    wheel_prob += self.pretrained_agents[agent]
                    if rand_choice < wheel_prob:
                        pretrained_choice = agent
                        n_old = np.random.randint(low=1, high=n_agents)
                        break
        n_new = n_agents - n_old
        versions, ratings = self._get_opponent_ids(n_new, n_old, pretrained_choice)
        agents = []
        for version in versions:
            if version == -1:
                agents.append(self.current_agent)
            elif pretrained_choice is not None and version == 'na':
                agents.append(pretrained_choice)
            else:
                selected_agent = self._get_past_model("-".join(version.split("-")[:-1]))
                if version.endswith("deterministic"):
                    selected_agent.deterministic = True
                elif version.endswith("stochastic"):
                    selected_agent.deterministic = False
                else:
                    raise ValueError("Unknown version type")
                agents.append(selected_agent)
            if self.streamer_mode > 1:
                agents[-1].deterministic = True
        versions = [v if v != -1 else latest_version for v in versions]
        return agents, pretrained_choice, versions, ratings
