# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from maro.communication import SessionMessage
from maro.rl.modeling_v2.ac_network import DiscreteVActorCriticNet
from maro.rl.utils import MsgKey, MsgTag, average_grads, discount_cumsum
from .buffer import Buffer
from .policy_base import RLPolicy


class ActorCritic(RLPolicy):
    def __init__(
        self,
        name: str,
        ac_net: DiscreteVActorCriticNet,
        reward_discount: float,
        grad_iters: int = 1,
        critic_loss_cls: Callable = None,
        min_logp: float = None,
        critic_loss_coef: float = 1.0,
        entropy_coef: float = .0,
        clip_ratio: float = None,
        lam: float = 0.9,
        max_trajectory_len: int = 10000,
        get_loss_on_rollout: bool = False,
        device: str = None
    ) -> None:
        super(ActorCritic, self).__init__(name, device)

        if not isinstance(ac_net, DiscreteVActorCriticNet):
            raise TypeError("model must be an instance of 'DiscreteVActorCriticNet'")

        self._ac_net = ac_net.to(self._device)
        self._reward_discount = reward_discount
        self._grad_iters = grad_iters
        self._critic_loss_func = critic_loss_cls if critic_loss_cls is not None else torch.nn.MSELoss
        self._min_logp = min_logp
        self._critic_loss_coef = critic_loss_coef
        self._entropy_coef = entropy_coef
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._max_trajectory_len = max_trajectory_len
        self._get_loss_on_rollout = get_loss_on_rollout

        self._buffer = defaultdict(lambda: Buffer(state_dim=self._ac_net.state_dim, size=self._max_trajectory_len))

    def __call__(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a list of action information dict given a batch of states.

        An action information dict contains the action itself, the corresponding log-P value and the corresponding
        state value.
        """
        self._ac_net.eval()
        states = torch.from_numpy(states).to(self._device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        with torch.no_grad():
            values = self._ac_net.get_values(states)
            if self._greedy:
                actions, logps = self._ac_net.get_actions_and_logps_greedy(states)
            else:
                actions, logps = self._ac_net.get_actions_and_logps(states)
        actions, logps, values = actions.cpu().numpy(), logps.cpu().numpy(), values.cpu().numpy()
        return actions, logps, values

    def data_parallel(self, *args, **kwargs) -> None:
        raise NotImplementedError  # TODO

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
            is True or the latest trajectory segment with pre-computed return and advantage values.
        """
        if self._get_loss_on_rollout:
            return self.get_batch_loss(self._get_batch(), explicit_grad=True)
        else:
            return self._get_batch()

    def _get_batch(self) -> dict:
        batch = defaultdict(list)
        for buf in self._buffer.values():
            trajectory = buf.get()
            values = np.append(trajectory["values"], trajectory["last_value"])
            rewards = np.append(trajectory["rewards"], trajectory["last_value"])
            deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]
            batch["states"].append(trajectory["states"])
            batch["actions"].append(trajectory["actions"])
            # Returns rewards-to-go, to be targets for the value function
            batch["returns"].append(discount_cumsum(rewards, self._reward_discount)[:-1])
            # Generalized advantage estimation using TD(Lambda)
            batch["advantages"].append(discount_cumsum(deltas, self._reward_discount * self._lam))
            batch["logps"].append(trajectory["logps"])

        return {key: np.concatenate(vals) for key, vals in batch.items()}

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        """Compute AC loss for a data batch.

        Args:
            batch (dict): A batch containing "states", "actions", "logps", "returns" and "advantages" as keys.
            explicit_grad (bool): If True, the gradients should be returned as part of the loss information. Defaults
                to False.
        """
        self._ac_net.train()
        states = torch.from_numpy(batch["states"]).to(self._device)
        actions = torch.from_numpy(batch["actions"]).to(self._device)
        logp_old = torch.from_numpy(batch["logps"]).to(self._device)
        returns = torch.from_numpy(batch["returns"]).to(self._device)
        advantages = torch.from_numpy(batch["advantages"]).to(self._device)

        action_probs = self._ac_net.get_probs(states)
        state_values = self._ac_net.get_values(states)
        state_values = state_values.squeeze()

        # actor loss
        logp = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())  # (N,)
        logp = torch.clamp(logp, min=self._min_logp, max=.0)
        if self._clip_ratio is not None:
            ratio = torch.exp(logp - logp_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        else:
            actor_loss = -(logp * advantages).mean()

        # critic_loss
        critic_loss = self._critic_loss_func(state_values, returns)
        # entropy
        entropy = -Categorical(action_probs).entropy().mean() if self._entropy_coef else 0

        # total loss
        loss = actor_loss + self._critic_loss_coef * critic_loss + self._entropy_coef * entropy

        loss_info = {
            "actor_loss": actor_loss.detach().cpu().numpy(),
            "critic_loss": critic_loss.detach().cpu().numpy(),
            "entropy": entropy.detach().cpu().numpy() if self._entropy_coef else .0,
            "loss": loss.detach().cpu().numpy() if explicit_grad else loss
        }
        if explicit_grad:
            loss_info["grad"] = self._ac_net.get_gradients(loss)

        return loss_info

    def update(self, loss_info_list: List[dict]) -> None:
        """Update the model parameters with gradients computed by multiple roll-out instances or gradient workers.

        Args:
            loss_info_list (List[dict]): A list of dictionaries containing loss information (including gradients)
                computed by multiple roll-out instances or gradient workers.
        """
        self._ac_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))

    def learn(self, batch: dict) -> None:
        """Learn from a batch containing data required for policy improvement.

        Args:
            batch (dict): A batch containing "states", "actions", "logps", "returns" and "advantages" as keys.
        """
        for _ in range(self._grad_iters):
            self._ac_net.step(self.get_batch_loss(batch)["loss"])

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
                            loss_info_by_policy[policy_name].append(loss_info)
                        else:
                            raise TypeError(f"Wrong type of loss_info: {type(loss_info)}")
                    dones += 1
                    if dones == len(msg_dict):
                        break
            # build dummy computation graph by `get_batch_loss` before apply gradients.
            _ = self.get_batch_loss(sub_batch, explicit_grad=True)
            self.update(loss_info_by_policy[self._name])

    def get_state(self) -> object:
        return self._ac_net.get_state()

    def set_state(self, state) -> None:
        self._ac_net.set_state(state)

    def load(self, path: str) -> None:
        """Load the policy state from disk."""
        self._ac_net.set_state(torch.load(path))

    def save(self, path: str) -> None:
        """Save the policy state to disk."""
        torch.save(self._ac_net.get_state(), path)
