import collections
from dataclasses import dataclass
from typing import Deque

import numpy as np


class BufferOld:
    """Store a sequence of transitions, i.e., a trajectory.

    Args:
        state_dim (int): State vector dimension.
        size (int): Buffer capacity, i.e., the maximum number of stored transitions.
    """

    def __init__(self, state_dim: int, size: int = 10000) -> None:
        self._states = np.zeros((size, state_dim), dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.int)
        self._logps = np.zeros(size, dtype=np.float32)
        self._values = np.zeros(size, dtype=np.float32)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.bool)
        self._size = size
        self._ptr = 0
        self._prev_ptr = 0

    def put(self, state: np.ndarray, action: dict, reward: float, terminal: bool = False) -> None:
        self._states[self._ptr] = state
        self._actions[self._ptr] = action.get("action", 0)
        self._logps[self._ptr] = action.get("logp", 0.0)
        self._values[self._ptr] = action.get("value", 0.0)
        self._rewards[self._ptr] = reward
        self._terminals[self._ptr] = terminal
        # increment pointer
        self._ptr += 1
        self._ptr %= self._size

    def get(self) -> dict:
        """Retrieve the latest trajectory segment."""
        terminal = self._terminals[self._ptr - 1]
        last = self._ptr - (not terminal)
        if last > self._prev_ptr:
            trajectory_slice = np.arange(self._prev_ptr, last)
        else:  # wrap-around
            trajectory_slice = np.concatenate([np.arange(self._prev_ptr, self._size), np.arange(last)])
        self._prev_ptr = last
        return {
            "states": self._states[trajectory_slice],
            "actions": self._actions[trajectory_slice],
            "logps": self._logps[trajectory_slice],
            "values": self._values[trajectory_slice],
            "rewards": self._rewards[trajectory_slice],
            "last_value": self._values[last]  # TODO: feature or bug?
        }


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
        assert state.shape == (1, self._state_dim)
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


if __name__ == '__main__':
    import time
    data = []
    for i in range(8000):
        element = BufferElement(
            np.random.uniform(size=(1, 10)).astype(np.float32),
            0,
            0.0,
            0.0,
            0.0,
            i % 10 == 9
        )
        data.append(element)

    b1 = BufferOld(state_dim=10, size=10000)
    b2 = Buffer(state_dim=10, size=10000)

    # for e in data[:10]:
    #     b1.put(e.state, {"action": e.action, "logp": e.logp, "value": e.value}, e.reward, e.terminal)
    #     b2.put(e.state, {"action": e.action, "logp": e.logp, "value": e.value}, e.reward, e.terminal)
    # print(b1.get())
    # print(b2.get())

    stime = time.perf_counter()
    for _ in range(100):
        for i in range(80):
            for e in data[i * 80: i * 80 + 80]:
                b1.put(e.state, {"action": e.action, "logp": e.logp, "value": e.value}, e.reward, e.terminal)
            b1.get()
    etime = time.perf_counter()
    print(etime - stime)

    stime = time.perf_counter()
    for _ in range(100):
        for i in range(80):
            for e in data[i * 80: i * 80 + 80]:
                b2.put(e.state, {"action": e.action, "logp": e.logp, "value": e.value}, e.reward, e.terminal)
            b2.get()
    etime = time.perf_counter()
    print(etime - stime)
