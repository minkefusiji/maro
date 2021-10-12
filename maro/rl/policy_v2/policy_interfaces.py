from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class DiscreteInterface:
    @abstractmethod
    def action_num(self) -> int:
        pass

    def action_scope(self) -> list:
        return list(range(self.action_num()))


class ContinuousInterface:
    @abstractmethod
    def action_range(self) -> Tuple[float, float]:  # TODO: `action_range` might not be a range.
        pass


class QNetworkInterface:
    @abstractmethod
    def q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        [batch_size, state_dim] + [batch_size, action_dim] => [batch_size, 1]
        """
        pass


class DiscreteQNetworkInterface(QNetworkInterface):
    @abstractmethod
    def q_values_for_all_actions(self, states: np.ndarray) -> np.ndarray:
        """
        [batch_size, state_dim] => [batch_size, action_num]
        """
        pass


class VNetworkInterface:
    @abstractmethod
    def v_values(self, states: np.ndarray) -> np.ndarray:
        """
        [batch_size, state_dim] => [batch_size, 1]
        """
        pass


class ValueBasedInterface(DiscreteQNetworkInterface, ABC):  # TODO: Might not be necessary for now.
    pass


class PolicyGradientInterface(ABC):  # TODO: Might not be necessary for now.
    pass
