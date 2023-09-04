from typing import Dict, Optional
from typing_extensions import TypedDict

from rocket_learn.agent.pretrained_policy import HardcodedAgent


class PretrainedAgent(TypedDict):
    prob: float  # Probability agent appears in training
    eval: bool  # Whether or not to include in eval pools
    # Probability of using deterministic in training, defaults to True
    p_deterministic_training: Optional[float]
    key: str  # The key to be used for the redis hash set, should be unique


PretrainedAgents = Dict[HardcodedAgent, PretrainedAgent]
