import collections
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass
class BufferElement:
    state: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    terminal: bool


class Buffer:
    """Store a sequence of transitions, i.e., a trajectory.

    Args:
        state_dim (int): State vector dimension.
        size (int): Buffer capacity, i.e., the maximum number of stored transitions.
    """
    def __init__(self, state_dim: int, size: int = 10000) -> None:
        self._pool: Deque[BufferElement] = collections.deque()
        self._state_dim = state_dim
        self._size = size

    def put(self, state: np.ndarray, action: dict, reward: float, terminal: bool = False) -> None:
        state = state.reshape(1, -1)
        self._pool.append(
            BufferElement(
                state,
                action.get("action", 0),
                action.get("logp", 0.0),
                action.get("value", 0.0),
                reward,
                terminal
            )
        )
        if len(self._pool) > self._size:
            self._pool.popleft()
            # TODO: erase the older elements or raise MLE error?

    def get(self) -> dict:
        """Retrieve the latest trajectory segment."""
        if len(self._pool) == 0:
            return {}

        if self._pool[-1].terminal:
            new_pool = collections.deque()
        else:
            new_pool = collections.deque()
            new_pool.append(self._pool.pop())

        ret = {
            "states": np.concatenate([elem.state for elem in self._pool], axis=0),
            "actions": np.array([elem.action for elem in self._pool], dtype=np.int32),
            "logps": np.array([elem.logp for elem in self._pool], dtype=np.float32),
            "values": np.array([elem.value for elem in self._pool], dtype=np.float32),
            "rewards": np.array([elem.reward for elem in self._pool], dtype=np.float32),
            "last_value": self._pool[-1].value
        }

        self._pool = new_pool
        return ret
