from abc import abstractmethod
from typing import Tuple

import torch

from maro.rl.modeling_v2.base_model import DiscretePolicyNetworkInterface, PolicyNetwork


class QNetwork(PolicyNetwork):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(QNetwork, self).__init__(state_dim, action_dim)

    @abstractmethod
    def q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return the Q-values according to states and actions.

        Args:
            states: shape = [batch_size, state_dim].
            actions: shape = [batch_size, action_dim].

        Returns:
            Q-values of shape [batch_size].
        """
        pass


class DiscreteQNetwork(QNetwork, DiscretePolicyNetworkInterface):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteQNetwork, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    def q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        [batch_size, state_dim] + [batch_size, action_dim] => [batch_size, 1]
        """
        q_matrix = self.q_values_for_all_actions(states)  # [batch_size, action_num]
        return q_matrix.gather(dim=1, index=actions).reshape(-1)

    @abstractmethod
    def q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return the matrix that contains Q-values for all possible actions.

        Args:
            states: shape = [batch_size, state_dim]

        Returns:
            Q-value matrix of shape [batch_size, action_num]
        """
        pass

    def get_actions_and_logps(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def action_num(self) -> int:
        return self._action_num

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_actions_greedy(self, states: torch.Tensor) -> torch.Tensor:
        return self.get_actions_and_logps_greedy(states)[0]
