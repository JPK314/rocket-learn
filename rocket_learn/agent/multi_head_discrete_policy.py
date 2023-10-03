from typing import Optional, List, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from rocket_learn.agent.policy import Policy
from rocket_learn.agent.aux_head import AuxHead


class MultiHeadDiscretePolicy(Policy):
    def __init__(self, body: nn.Module, head: nn.Module, aux_heads: Tuple[AuxHead], shape: Tuple[int, ...] = (3,) * 5 + (2,) * 3, deterministic=False):

        super().__init__(deterministic)
        self.body = body
        self.head = head
        self.aux_heads = aux_heads
        self.shape = shape

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        elif isinstance(obs, tuple):
            obs = tuple(o if isinstance(o, th.Tensor) else th.from_numpy(o).float() for o in obs)
        return self.body(obs)

    def get_aux_heads_labels(self, rewards, game_states, car_ids):
        return tuple(aux_head.get_label(rewards, game_states, car_ids) for aux_head in self.aux_heads)

    def get_aux_head_predictions(self, body_out):
        return tuple(aux_head.get_prediction(body_out) for aux_head in self.aux_heads)
    
    def get_aux_head_losses(self, labels, predictions):
        """
        :param labels: tuple of tensors, where each tensor corresponds to a list of labels formed from the game states that led to observations, and each element in the tuple is an aux head
        :param predictions: tuple of tuples of prediction tensors - the inner tuple is in parallel with the corresponding label tensor, and each element in the outer tuple corresponds to an aux head
        """

        return tuple(th.mean(aux_head.grade_prediction(labels[i], predictions[i])) for i,aux_head in enumerate(self.aux_heads))

    def get_action_distribution(self, body_out):
        logits = self.head(body_out)
        return Categorical(logits=logits)

    def sample_action(
            self,
            distribution: Categorical,
            deterministic=None
    ):
        if deterministic is None:
            deterministic = self.deterministic
        if deterministic:
            action_indices = th.argmax(distribution.logits, dim=-1)
        else:
            action_indices = distribution.sample()

        return action_indices

    def log_prob(self, distribution: Categorical, selected_action):
        log_prob = distribution.log_prob(selected_action).sum(dim=-1)
        return log_prob

    def entropy(self, distribution: Categorical, selected_action):
        entropy = distribution.entropy().sum(dim=-1)
        return entropy

    def env_compatible(self, action):
        if isinstance(action, th.Tensor):
            action = action.numpy()
        return action