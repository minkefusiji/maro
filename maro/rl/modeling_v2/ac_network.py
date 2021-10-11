from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.distributions import Categorical

from .base_model import DiscretePolicyNetworkInterface, PolicyNetwork


class ActorCriticCoreModel(PolicyNetwork, ABC):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(ActorCriticCoreModel, self).__init__(state_dim, action_dim)


class QCriticInterface:
    @abstractmethod
    def q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        [batch_size, state_dim] + [batch_size, action_dim] => [batch_size]
        """
        pass


class VCriticInterface:
    @abstractmethod
    def v_critic(self, states: torch.Tensor) -> torch.Tensor:
        """
        [batch_size, state_dim] => [batch_size]
        """
        pass


class DiscreteActorCriticNet(ActorCriticCoreModel, DiscretePolicyNetworkInterface, ABC):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteActorCriticNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    @property
    def action_num(self) -> int:
        return self._action_num

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self.get_actions_and_logps(states)[0]

    def get_actions_greedy(self, states: torch.Tensor) -> torch.Tensor:
        return self.get_actions_and_logps_greedy(states)[0]


class DiscreteQActorCriticNet(DiscreteActorCriticNet, QCriticInterface):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteQActorCriticNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    def q_critic(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        [batch_size, state_dim] + [batch_size, 1] => [batch_size, 1]
        """
        q_matrix = self.q_critic_for_all_actions(states)  # [batch_size, action_num]
        return q_matrix.gather(dim=1, index=actions).reshape(-1)

    @abstractmethod
    def q_critic_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        [batch_size, state_dim] => [batch_size, action_num]
        """
        pass


class DiscreteVActorCriticNet(DiscreteActorCriticNet, VCriticInterface):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(DiscreteVActorCriticNet, self).__init__(state_dim=state_dim, action_dim=1)
        self._action_num = action_num

    def get_actions_and_logps(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = Categorical(self.get_probs(states))
        actions = action_probs.sample()
        logps = action_probs.log_prob(actions)
        return actions, logps

    @abstractmethod
    def get_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        [batch_size, state_dim] => [batch_size, 1]
        """
        pass
