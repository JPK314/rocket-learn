import itertools
from collections import Counter
from typing import Iterator, Callable, Optional, List

import numpy as np
import plotly.graph_objs as go
# import matplotlib.pyplot  # noqa
import wandb
# from matplotlib.axes import Axes
# from matplotlib.figure import Figure
from redis import Redis
from redis.exceptions import ResponseError
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.action_parsers import ActionParser
from trueskill import Rating, rate, SIGMA

from rocket_learn.agent.types import PretrainedAgents
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.rollout_generator.redis.utils import decode_buffers, _unserialize, PRETRAINED_QUALITIES, QUALITIES, _serialize, ROLLOUTS, \
    VERSION_LATEST, OPPONENT_MODELS, CONTRIBUTORS, N_UPDATES, MODEL_LATEST, _serialize_model, get_rating, get_ratings, \
    get_pretrained_rating, get_pretrained_ratings, add_pretrained_ratings, _ALL, LATEST_RATING_ID, EXPERIENCE_PER_MODE
from rocket_learn.utils.stat_trackers.stat_tracker import StatTracker


class RedisRolloutGenerator(BaseRolloutGenerator):
    """
    Rollout generator in charge of sending commands to workers via redis
    """

    def __init__(
            self,
            name: str,
            redis: Redis,
            obs_build_factory: Callable[[], ObsBuilder],
            rew_func_factory: Callable[[], RewardFunction],
            act_parse_factory: Callable[[], ActionParser],
            save_every=10,
            model_every=100,
            logger=None,
            clear=True,
            max_age=0,
            default_sigma=SIGMA,
            min_sigma=1,
            gamemodes=("1v1", "2v2", "3v3"),
            pretrained_agents: PretrainedAgents = None,
            stat_trackers: Optional[List[StatTracker]] = None,
    ):
        self.lastsave_ts = None
        self.name = name
        self.tot_bytes = 0
        self.redis = redis
        self.logger = logger
        self.pretrained_agents_keys = []
        if pretrained_agents is not None:
            add_pretrained_ratings(
                self.redis, pretrained_agents, gamemodes=gamemodes)
            for config in pretrained_agents.values():
                self.pretrained_agents_keys.append(config["key"])

        # TODO saving/loading
        if clear:
            self.redis.delete(*(_ALL + tuple(QUALITIES.format(gm)
                              for gm in gamemodes)))
            self.redis.set(N_UPDATES, 0)
        else:
            if self.redis.exists(ROLLOUTS) > 0:
                self.redis.delete(ROLLOUTS)
            self.redis.decr(VERSION_LATEST,
                            max_age + 1)  # In case of reload from old version, don't let current seep in
        self.redis.hset(EXPERIENCE_PER_MODE, mapping={m: 0 for m in gamemodes})

        # self.redis.set(SAVE_FREQ, save_every)
        # self.redis.set(MODEL_FREQ, model_every)
        self.save_freq = save_every
        self.model_freq = model_every
        self.contributors = Counter()  # No need to save, clears every iteration
        self.obs_build_func = obs_build_factory
        self.rew_func_factory = rew_func_factory
        self.act_parse_factory = act_parse_factory
        self.max_age = max_age
        self.default_sigma = default_sigma
        self.min_sigma = min_sigma
        self.gamemodes = gamemodes
        self.stat_trackers = stat_trackers or []
        self._reset_stats()

    @staticmethod
    def _process_rollout(rollout_bytes, latest_version, obs_build_func, rew_build_func, act_build_func, max_age):
        rollout_data, versions, uuid, name, result, has_obs, has_states, has_rewards = _unserialize(
            rollout_bytes)

        v_check = [v for v in versions if isinstance(
            v, int) or v.startswith("-")]

        if any(version < 0 and abs(version - latest_version) > max_age for version in v_check):
            return

        if any(version < 0 for version in v_check):
            buffers, states = decode_buffers(rollout_data, versions, has_obs, has_states, has_rewards,
                                             obs_build_func, rew_build_func, act_build_func)
        else:
            buffers = states = [None] * len(v_check)

        return buffers, states, versions, uuid, name, result

    def _update_ratings(self, name, versions, buffers, latest_version, result):
        ratings = []
        relevant_buffers = []
        # TODO: support unfair games
        gamemode = f"{len(versions) // 2}v{len(versions) // 2}"

        has_buffer_versions = [
            v for v in versions if isinstance(v, int) or (v != "human" and ("-".join(v.split("-")[:-1]) not in self.pretrained_agents_keys and v not in self.pretrained_agents_keys))]
        for version, buffer in itertools.zip_longest(has_buffer_versions, buffers):
            if isinstance(version, int) and version < 0:
                if abs(version - latest_version) <= self.max_age:
                    relevant_buffers.append(buffer)
                    self.contributors[name] += buffer.size()
                else:
                    return []

        rated_versions = [v for v in versions if isinstance(
            v, str) and v != "human"]
        for version in rated_versions:
            short_name = "-".join(version.split("-")[:-1])
            if version in self.pretrained_agents_keys or short_name in self.pretrained_agents_keys:
                rating = get_pretrained_rating(
                    gamemode, version, self.redis)
            else:
                rating = get_rating(gamemode, version, self.redis)
            ratings.append(rating)

        # Only old versions, calculate MMR
        if len(ratings) == len(versions) and len(buffers) == 0:
            blue_players = len(versions) // 2
            blue = tuple(ratings[:blue_players])  # Tuple is important
            orange = tuple(ratings[blue_players:])

            # In ranks lowest number is best, result=-1 is orange win, 0 tie, 1 blue
            r1, r2 = rate((blue, orange), ranks=(0, result))

            # Some trickery to handle same rating appearing multiple times, we just average their new mus and sigmas
            ratings_versions = {}
            for rating, version in zip(r1 + r2, versions):
                ratings_versions.setdefault(version, []).append(rating)

            mapping_past = {}
            mapping_pretrained = {}
            for version, ratings in ratings_versions.items():
                # In case of duplicates, average ratings together (not strictly necessary with default setup)
                # Also limit sigma to its lower bound
                avg_rating = Rating(sum(r.mu for r in ratings) / len(ratings),
                                    max(sum(r.sigma ** 2 for r in ratings) ** 0.5 / len(ratings), self.min_sigma))
                short_name = "-".join(version.split("-")[:-1])
                if version in self.pretrained_agents_keys or short_name in self.pretrained_agents_keys:
                    mapping_pretrained[version] = _serialize(tuple(avg_rating))
                else:
                    mapping_past[version] = _serialize(tuple(avg_rating))
            # TODO: support unfair games
            gamemode = f"{len(versions) // 2}v{len(versions) // 2}"

            if mapping_past:
                self.redis.hset(QUALITIES.format(
                    gamemode), mapping=mapping_past)
            if mapping_pretrained:
                self.redis.hset(PRETRAINED_QUALITIES.format(gamemode),
                                mapping=mapping_pretrained)

        if len(relevant_buffers) > 0:
            self.redis.hincrby(EXPERIENCE_PER_MODE, gamemode, len(
                relevant_buffers) * relevant_buffers[0].size())

        return relevant_buffers

    def _reset_stats(self):
        for stat_tracker in self.stat_trackers:
            stat_tracker.reset()

    def _update_stats(self, states, mask):
        if states is None:
            return
        for stat_tracker in self.stat_trackers:
            stat_tracker.update(states, mask)

    def _get_stats(self):
        stats = {}
        for stat_tracker in self.stat_trackers:
            stats[stat_tracker.name] = stat_tracker.get_stat()
        return stats

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            latest_version = int(self.redis.get(VERSION_LATEST))
            data = self.redis.blpop(ROLLOUTS)[1]
            self.tot_bytes += len(data)
            res = self._process_rollout(
                data, latest_version,
                self.obs_build_func, self.rew_func_factory, self.act_parse_factory,
                self.max_age
            )
            if res is not None:
                buffers, states, versions, uuid, name, result = res
                # versions = [version for version in versions if version != 'na']  # don't track humans or hardcoded

                relevant_buffers = self._update_ratings(
                    name, versions, buffers, latest_version, result)
                if len(relevant_buffers) > 0:
                    self._update_stats(
                        states, [b in relevant_buffers for b in buffers])
                yield from relevant_buffers

    def _plot_ratings(self):
        fig_data = []
        i = 0
        means = {}
        mean_key = "mean"
        gamemodes = list(self.gamemodes)
        if len(gamemodes) > 1:
            gamemodes.append(mean_key)
        for gamemode in gamemodes:
            if gamemode != mean_key:
                ratings = get_ratings(gamemode, self.redis)
                if len(ratings) <= 0:
                    return
            baseline = None
            for mode in "stochastic", "deterministic":
                if gamemode != "mean":
                    x = []
                    mus = []
                    sigmas = []
                    for k, r in ratings.items():  # noqa
                        if k.endswith(mode):
                            v = int(k.rsplit("-", 2)[1][1:])
                            # v = int(k.split("-")[1][1:])
                            x.append(v)
                            mus.append(r.mu)
                            sigmas.append(r.sigma)
                            mean = means.setdefault(mode, {}).get(v, (0, 0))
                            means[mode][v] = (
                                mean[0] + r.mu, mean[1] + r.sigma ** 2)
                            # *Smoothly* transition from red, to green, to blue depending on gamemode
                    mid = (len(self.gamemodes) - 1) / 2
                    # avoid divide by 0 issues if there's only one gamemode, this moves it halfway into the colors
                    if mid == 0:
                        mid = 0.5
                    if i < mid:
                        r = 1 - i / mid
                        g = i / mid
                        b = 0
                    else:
                        r = 0
                        g = 1 - (i - mid) / mid
                        b = (i - mid) / mid
                else:
                    means_mode = means.get(mode, {})
                    x = list(means_mode.keys())
                    mus = [mean[0] / len(self.gamemodes)
                           for mean in means_mode.values()]
                    sigmas = [(mean[1] / len(self.gamemodes)) **
                              0.5 for mean in means_mode.values()]
                    r = g = b = 1 / 3

                indices = np.argsort(x)
                x = np.array(x)[indices]
                mus = np.array(mus)[indices]
                sigmas = np.array(sigmas)[indices]

                if baseline is None:
                    # Stochastic initialization is defined as the baseline (0 mu)
                    baseline = mus[0]
                mus = mus - baseline
                y = mus
                y_upper = mus + 2 * sigmas  # 95% confidence
                y_lower = mus - 2 * sigmas

                scale = 255 if mode == "stochastic" else 128
                color = f"{int(r * scale)},{int(g * scale)},{int(b * scale)}"

                fig_data += [
                    go.Scatter(
                        x=x,
                        y=y,
                        line=dict(color=f'rgb({color})'),
                        mode='lines',
                        name=f"{gamemode}-{mode}",
                        legendgroup=f"{gamemode}-{mode}",
                        showlegend=True,
                        visible=None if gamemode == gamemodes[-1] else "legendonly",
                    ),
                    go.Scatter(
                        x=np.concatenate((x, x[::-1])),  # x, then x reversed
                        # upper, then lower reversed
                        y=np.concatenate((y_upper, y_lower[::-1])),
                        fill='toself',
                        fillcolor=f'rgba({color},0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        name="sigma",
                        legendgroup=f"{gamemode}-{mode}",
                        showlegend=False,
                        visible=None if gamemode == gamemodes[-1] else "legendonly",
                    ),
                ]
            i += 1

        if len(fig_data) <= 0:
            return

        fig = go.Figure(fig_data)
        fig.update_layout(
            title="Rating", xaxis_title="Iteration", yaxis_title="TrueSkill")

        self.logger.log({
            "qualities": fig,
        }, commit=False)

    def _add_opponent(self, agent):
        latest_id = self.redis.get(LATEST_RATING_ID)
        prefix = f"{self.name}-v"
        if latest_id is None:
            version = 0
        else:
            latest_id = latest_id.decode("utf-8")
            version = int(latest_id.replace(prefix, "")) + 1
        key = f"{prefix}{version}"

        # Add to list
        self.redis.hset(OPPONENT_MODELS, key, agent)

        # Set quality
        for gamemode in self.gamemodes:
            ratings = get_ratings(gamemode, self.redis)

            for mode in "stochastic", "deterministic":
                if latest_id is not None:
                    latest_key = f"{latest_id}-{mode}"
                    quality = Rating(
                        ratings[latest_key].mu, self.default_sigma)
                else:
                    # First (basically random) agent is initialized at 0
                    quality = Rating(0, self.min_sigma)

                self.redis.hset(QUALITIES.format(gamemode),
                                f"{key}-{mode}", _serialize(tuple(quality)))

        # Inform that new opponent is ready
        self.redis.set(LATEST_RATING_ID, key)

    def update_parameters(self, new_params):
        """
        update redis (and thus workers) with new model data and save data as future opponent
        :param new_params: new model parameters
        """
        model_bytes = _serialize_model(new_params)
        self.redis.set(MODEL_LATEST, model_bytes)
        self.redis.decr(VERSION_LATEST)

        print("Top contributors:\n" +
              "\n".join(f"\t{c}: {n}" for c, n in self.contributors.most_common(5)))
        self.logger.log({
            "redis/contributors": wandb.Table(columns=["name", "steps"], data=self.contributors.most_common())},
            commit=False
        )

        pretrained_qualities = {}
        for gamemode in self.gamemodes:
            non_pretrained_min_rating = min(
                [r.mu for r in get_ratings(gamemode, self.redis).values()])
            qualities = get_pretrained_ratings(gamemode, self.redis)
            pretrained_qualities[gamemode] = []
            for agent, rating in qualities.items():
                pretrained_qualities[gamemode].append(
                    (agent, rating.mu-non_pretrained_min_rating))

        for gamemode in self.gamemodes:
            self.logger.log({
                "pretrained/qualities-" + gamemode: wandb.Table(columns=["name", "rating"], data=pretrained_qualities[gamemode])
            }, commit=False)

        if self.gamemodes[0] != '1v0':
            self._plot_ratings()
        tot_contributors = self.redis.hgetall(CONTRIBUTORS)
        tot_contributors = Counter({name: int(count)
                                   for name, count in tot_contributors.items()})
        tot_contributors += self.contributors
        if tot_contributors:
            self.redis.hset(CONTRIBUTORS, mapping=tot_contributors)
        self.contributors.clear()
        stat_keys = self.redis.keys("selector_stat*")
        for key in stat_keys:
            value = self.redis.get(key)
            if value is not None:
                value = float(value)
                value = value / 2
                self.redis.set(key, str(int(value)))
        self.logger.log({"redis/rollout_bytes": self.tot_bytes}, commit=False)
        self.tot_bytes = 0

        n_updates = self.redis.incr(N_UPDATES) - 1
        # save_freq = int(self.redis.get(SAVE_FREQ))

        if n_updates > 0:
            self.logger.log({f"stat/{name}": value for name,
                            value in self._get_stats().items()}, commit=False)
        self._reset_stats()

        if n_updates % self.model_freq == 0:
            print("Adding model to pool...")
            self._add_opponent(model_bytes)

        if n_updates % self.save_freq == 0:
            # self.redis.set(MODEL_N.format(self.n_updates // self.save_every), model_bytes)
            print("Saving model...")
            if self.lastsave_ts == self.redis.lastsave():
                print("redis save error, previous bgsave failed")
            self.lastsave_ts = self.redis.lastsave()
            try:
                self.redis.bgsave()
            except ResponseError:
                print("redis bgsave failed, auto save already in progress")
