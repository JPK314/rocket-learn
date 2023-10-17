from abc import abstractmethod
from typing import Optional, List, Tuple

import numpy as np
import torch as th
from torch import nn

class AuxHead(nn.Module):
    def __init__(self, net: nn.Module, deterministic=False):
        super().__init__()
        self.net = net
        self.deterministic = deterministic

    def forward(self, obs):
        logits = self.net(obs)
        return logits

    def get_prediction(self, body_out):
        """
        Defines how the output of the shared body turns into a prediction.
        :param body_out: Tensor where 0th dimension is batching dimension, and other dimensions form the tensor for a single body output.
        
        :return: Tuple of torch Tensors.
        """
        return self(body_out)

    @abstractmethod
    def get_label(self, rewards, game_states, car_ids, observations):
        """
        Defines the label that the head will try to predict.
        :param rewards: numpy array indexed first by agent and second in parallel with game states.
        :param game_states: list of game states in the order they occurred in the trajectory.
        :param car_ids: list of car ids in parallel with the first index of the rewards array.
        :param observations: list of lists of observations, indexed first by agent and second in parallel with game states.

        :return: numpy array of labels, indexed first by agent and second in parallel with game states.
        """
        raise NotImplementedError

    @abstractmethod
    def grade_prediction(self, labels, predictions):
        """
        Defines how a label and a prediction turn into a loss term.
        :param labels: torch Tensor of labels (th.from_numpy(numpy_array)).
        :param predictions: Tensor where 0th dimension is parallel with labels, and the other dimensions form the tensor for the head output.

        :return: torch Tensor of grades, or tuple of torch Tensors of grades.
        """
        raise NotImplementedError
