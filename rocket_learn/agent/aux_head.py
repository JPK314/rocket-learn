from abc import abstractmethod
from typing import Optional, List, Tuple

import numpy as np
import torch as th
from torch import nn

class AuxHead(nn.Module):
    def __init__(self, net: nn.Module, shape: Tuple[int, ...] = (3,) * 5 + (2,) * 3, deterministic=False):
        super().__init__()
        self.net = net
        self.shape = shape
        self.deterministic = deterministic

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    @abstractmethod
    def get_label(self, rewards, game_states, car_ids):
        """
        Defines the label that the head will try to predict.
        :param rewards: numpy array indexed first by agent and second in parallel with game states.
        :param game_states: list of game states in the order they occurred in the trajectory.
        :param car_ids: list of car ids in parallel with the first index of the rewards array.

        :return: numpy array of labels, indexed first by agent and second in parallel with game states.
        """
        raise NotImplementedError

    @abstractmethod
    def get_prediction(self, body_out):
        """
        Defines how the output of the shared body turns into a prediction.
        :param body_out: Tuple of torch Tensors, where each Tensor is the output of the shared body for an agent.
        
        :return: Tuple of torch Tensors.
        """
        return self(body_out)

    def grade_prediction(self, labels, predictions):
        """
        Defines how a label and a prediction turn into a loss term.
        :param labels: torch Tensor of labels (th.from_numpy(numpy_array)).
        :param predictions: Tuple of Torch tensors, where each Tensor corresponds to an element of the labels Tensor.

        :return: torch Tensor of grades
        """
        raise NotImplementedError
