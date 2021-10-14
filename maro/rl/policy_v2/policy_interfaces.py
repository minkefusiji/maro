from abc import abstractmethod
from typing import Tuple

import numpy as np


"""
Mixins for policies.

Mixins have only methods (abstract or non abstract) which define a set of functions that a type of policies should have.
Abstract methods should be implemented by lower-level mixins or policy classes that inherit the mixin.

A policy class could inherit multiple mixins so that the combination of mixins determines the entire set of methods
of this policy.
"""


class DiscreteActionMixin:
    """Mixin for policies that generate discrete actions.
    """
    @abstractmethod
    def action_num(self) -> int:
        pass


class ContinuousActionMixin:
    """Mixin for policies that generate continuous actions.
    """
    @abstractmethod
    def action_range(self) -> Tuple[float, float]:  # TODO: `action_range` might not be a range.
        """Returns The value range of the action: [lower, upper] (inclusive).
        """
        pass


class QNetworkMixin:
    """Mixin for policies that have a Q-network in it, no matter how it is used. For example,
    both DQN policies and Actor-Critic policies that use a Q-network as the critic should inherit this mixin.
    """
    @abstractmethod
    def q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Returns Q-values based on given states and actions.

        Args:
            states (np.ndarray): [batch_size, state_dim]
            actions (np.ndarray): [batch_size, action_dim]

        Returns:
            Q-values (np.ndarray): [batch_size, 1]
        """
        pass


class DiscreteQNetworkMixin(DiscreteActionMixin, QNetworkMixin):
    """Combination of DiscreteActionMixin and QNetworkMixin.
    """
    @abstractmethod
    def q_values_for_all_actions(self, states: np.ndarray) -> np.ndarray:
        """Returns Q-values for all actions based on given states

        Args:
            states (np.ndarray): [batch_size, state_dim]

        Returns:
            Q-values (np.ndarray): [batch_size, action_num]
        """
        pass


class VNetworkMixin:
    """Mixin for policies that have a V-network in it. Similar to QNetworkMixin.
    """
    @abstractmethod
    def v_values(self, states: np.ndarray) -> np.ndarray:
        """Returns Q-values based on given states.

        Args:
            states (np.ndarray): [batch_size, state_dim]

        Returns:
            V-values (np.ndarray): [batch_size, 1]
        """
        pass
