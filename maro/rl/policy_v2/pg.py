# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch

from maro.communication import SessionMessage
from maro.rl.modeling_v2.pg_network import DiscretePolicyGradientNetwork
from maro.rl.utils import MsgKey, MsgTag, average_grads, discount_cumsum
from .buffer import Buffer
from .policy_base import RLPolicy
from .policy_interfaces import DiscreteInterface


class DiscretePolicyGradient(DiscreteInterface, RLPolicy):
    def __init__(
        self,
        name: str,
        policy_net: DiscretePolicyGradientNetwork,
        reward_discount: float,
        grad_iters: int = 1,
        max_trajectory_len: int = 10000,
        get_loss_on_rollout: bool = False,
        device: str = None
    ) -> None:
        super(DiscretePolicyGradient, self).__init__(name=name, device=device)

        if not isinstance(policy_net, DiscretePolicyGradientNetwork):
            raise TypeError("model must be an instance of 'DiscretePolicyGradientNetwork'")

        self._policy_net = policy_net.to(self._device)
        self._reward_discount = reward_discount
        self._grad_iters = grad_iters
        self._max_trajectory_len = max_trajectory_len
        self._get_loss_on_rollout = get_loss_on_rollout

        self._buffer = defaultdict(lambda: Buffer(state_dim=self._policy_net.state_dim, size=self._max_trajectory_len))

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """Return a list of action information dict given a batch of states.

        An action information dict contains the action itself and the corresponding log-P value.
        """
        return self.get_actions_with_logps(states)[0]

    def get_actions_with_logps(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._policy_net.eval()
        with torch.no_grad():
            states: torch.Tensor = torch.from_numpy(states).to(self._device)
            if not self._in_exploration_mode:
                actions, logps = self._policy_net.get_actions_and_logps_exploitation(states)
            else:
                actions, logps = self._policy_net.get_actions_and_logps_exploration(states)
        actions, logps = actions.cpu().numpy(), logps.cpu().numpy()
        return actions, logps

    def data_parallel(self, *args, **kwargs) -> None:
        raise NotImplementedError  # TODO

    def action_num(self) -> int:
        return self._policy_net.action_num()

    def record(
        self,
        key: str,
        state: np.ndarray,
        action: dict,
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ) -> None:
        self._buffer[key].put(state, action, reward, terminal)

    def get_rollout_info(self) -> dict:
        """Extract information from the recorded transitions.

        Returns:
            Loss (including gradients) for the latest trajectory segment in the replay buffer if ``get_loss_on_rollout``
            is True or the latest trajectory segment with pre-computed return values.
        """
        if self._get_loss_on_rollout:
            return self.get_batch_loss(self._get_batch(), explicit_grad=True)
        else:
            return self._get_batch()

    def _get_batch(self) -> dict:
        batch = defaultdict(list)
        for buf in self._buffer:
            trajectory = buf.get()
            rewards = np.append(trajectory["rewards"], trajectory["last_val"])
            batch["states"].append(trajectory["states"])
            # Returns rewards-to-go, to be targets for the value function
            batch["returns"].append(discount_cumsum(rewards, self._reward_discount)[:-1])

        return {key: np.concatenate(vals) for key, vals in batch.items()}

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        """Compute AC loss for a data batch.

        Args:
            batch (dict): A batch containing "states" and "returns" as keys.
            explicit_grad (bool): If True, the gradients should be returned as part of the loss information. Defaults
                to False.
        """
        self._policy_net.train()
        returns = torch.from_numpy(np.asarray(batch["returns"])).to(self._device)

        logps = self._policy_net.get_logps(batch["states"])
        loss = -(logps * returns).mean()
        loss_info = {"loss": loss.detach().cpu().numpy() if explicit_grad else loss}
        if explicit_grad:
            loss_info["grad"] = self._policy_net.get_gradients(loss)
        return loss_info

    def update(self, loss_info_list: List[dict]) -> None:
        """Update the model parameters with gradients computed by multiple roll-out instances or gradient workers.

        Args:
            loss_info_list (List[dict]): A list of dictionaries containing loss information (including gradients)
                computed by multiple roll-out instances or gradient workers.
        """
        self._policy_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))

    def learn(self, batch: dict) -> None:
        """Learn from a batch containing data required for policy improvement.

        Args:
            batch (dict): A batch containing "states" and "returns" as keys.
        """
        for _ in range(self._grad_iters):
            self._policy_net.step(self.get_batch_loss(batch)["grad"])

    def improve(self) -> None:
        """Learn using data from the buffer."""
        self.learn(self._get_batch())

    def learn_with_data_parallel(self, batch: dict, worker_id_list: list) -> None:
        assert hasattr(self, '_proxy'), "learn_with_data_parallel is invalid before data_parallel is called."

        for _ in range(self._grad_iters):
            msg_dict = defaultdict(lambda: defaultdict(dict))
            sub_batch = {}
            for i, worker_id in enumerate(worker_id_list):
                sub_batch = {key: batch[key][i::len(worker_id_list)] for key in batch}
                msg_dict[worker_id][MsgKey.GRAD_TASK][self._name] = sub_batch
                msg_dict[worker_id][MsgKey.POLICY_STATE][self._name] = self.get_state()
                # data-parallel
                self._proxy.isend(SessionMessage(
                    MsgTag.COMPUTE_GRAD, self._proxy.name, worker_id, body=msg_dict[worker_id]))
            dones = 0
            loss_info_by_policy = {self._name: []}
            for msg in self._proxy.receive():
                if msg.tag == MsgTag.COMPUTE_GRAD_DONE:
                    for policy_name, loss_info in msg.body[MsgKey.LOSS_INFO].items():
                        if isinstance(loss_info, list):
                            loss_info_by_policy[policy_name] += loss_info
                        elif isinstance(loss_info, dict):
                            loss_info_by_policy[policy_name].append(loss_info["grad"])
                        else:
                            raise TypeError(f"Wrong type of loss_info: {type(loss_info)}")
                    dones += 1
                    if dones == len(msg_dict):
                        break
            # build dummy computation graph before apply gradients.
            _ = self.get_batch_loss(sub_batch, explicit_grad=True)
            self._policy_net.step(loss_info_by_policy[self._name])

    def get_state(self) -> object:
        return self._policy_net.get_state()

    def set_state(self, state) -> None:
        self._policy_net.set_state(state)

    def load(self, path: str) -> None:
        """Load the policy state from disk."""
        self._policy_net.set_state(torch.load(path))

    def save(self, path: str) -> None:
        """Save the policy state to disk."""
        torch.save(self._policy_net.get_state(), path)
