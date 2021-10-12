from abc import ABC
from typing import Tuple

import torch
from torch.distributions import Categorical

from .base_model import DiscretePolicyNetworkInterface, PolicyNetwork


class PolicyGradientNetwork(PolicyNetwork, ABC):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(PolicyGradientNetwork, self).__init__(state_dim, action_dim)


class DiscretePolicyGradientNetwork(DiscretePolicyNetworkInterface, ABC, PolicyGradientNetwork):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscretePolicyGradientNetwork, self).__init__(state_dim, 1)
        self._action_num = action_num

    def get_actions_and_logps_exploration(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = Categorical(self.get_probs(states))
        actions = action_probs.sample()
        logps = action_probs.log_prob(actions)
        return actions, logps

    @property
    def action_num(self) -> int:
        return self._action_num

    def get_actions_exploration(self, states: torch.Tensor) -> torch.Tensor:
        return self.get_actions_and_logps_exploration(states)[0]

    def get_actions_exploitation(self, states: torch.Tensor) -> torch.Tensor:
        return self.get_actions_and_logps_greedy(states)[0]
