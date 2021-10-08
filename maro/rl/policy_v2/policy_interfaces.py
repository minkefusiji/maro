from abc import ABC, abstractmethod
from typing import List, Tuple


class DiscreteInterface:
    def action_num(self) -> int:
        return len(self.action_scope())

    @abstractmethod
    def action_scope(self) -> set:
        pass


class ContinuousInterface:
    @abstractmethod
    def action_range(self) -> Tuple[float, float]:  # TODO: `action_range` might not be a range.
        pass


class QNetworkInterface:
    @abstractmethod
    def q_value(self, state: object, action: object) -> float:
        pass


class DiscreteQNetworkInterface(QNetworkInterface):
    @abstractmethod
    def q_values_for_all_actions(self, state: object) -> List[float]:
        pass


class VNetWorkInterface:
    @abstractmethod
    def v_value(self, state: object) -> float:
        pass


class ValueBasedInterface(DiscreteQNetworkInterface, ABC):  # TODO: Might not be necessary for now.
    pass


class PolicyGradientInterface(ABC):  # TODO: Might not be necessary for now.
    pass
