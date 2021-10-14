# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Tuple

import torch


class AbsCoreModel(torch.nn.Module):
    """TODO
    """
    def __init__(self):
        super(AbsCoreModel, self).__init__()

    @abstractmethod
    def step(self, loss: torch.tensor) -> None:
        """Use a computed loss to back-propagate gradients and apply them to the underlying parameters.

        Args:
            loss: Result of a computation graph that involves the underlying parameters.
        """
        pass

    @abstractmethod
    def get_gradients(self, loss: torch.tensor) -> torch.tensor:
        """Get gradients from a computed loss.

        There are two possible scenarios where you need to implement this interface: 1) if you are doing distributed
        learning and want each roll-out instance to collect gradients that can be directly applied to policy parameters
        on the learning side (abstracted through ``AbsPolicyManager``); 2) if you are computing loss in data-parallel
        fashion, i.e., by splitting a data batch to several smaller batches and sending them to a set of remote workers
        for parallelized gradient computation. In this case, this method will be used by the remote workers.
        """
        pass

    @abstractmethod
    def apply_gradients(self, grad: dict) -> None:
        """Apply gradients to the model parameters.

        This needs to be implemented together with ``get_gradients``.
        """
        pass

    @abstractmethod
    def get_state(self) -> object:
        """Return the current model state.

        Ths model state usually involves the "state_dict" of the module as well as those of the embedded optimizers.
        """
        pass

    @abstractmethod
    def set_state(self, state: object) -> None:
        """Set model state.

        Args:
            state: Model state to be applied to the instance. Ths model state is either the result of a previous call
            to ``get_state`` or something loaded from disk and involves the "state_dict" of the module as well as those
            of the embedded optimizers.
        """
        pass

    def soft_update(self, other_model: torch.nn.Module, tau: float) -> None:
        """Soft-update model parameters using another model.

        Update formulae: param = (1 - tau) * param + tau * other_param.

        Args:
            other_model: The model to update the current model with.
            tau (float): Soft-update coefficient.
        """
        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data


class PolicyNetwork(AbsCoreModel):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(PolicyNetwork, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @abstractmethod
    def get_actions_exploration(self, states: torch.Tensor) -> torch.Tensor:
        """Get actions under exploration mode.
        """
        pass

    @abstractmethod
    def get_actions_exploitation(self, states: torch.Tensor) -> torch.Tensor:
        """Get actions under exploitation mode.
        """
        pass


class DiscretePolicyNetworkMixin:
    @abstractmethod
    def get_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get probabilities of all possible actions.

        Args:
            states: [batch_size, state_dim]

        Returns:
            probability matrix: [batch_size, action_num]
        """
        pass

    def get_logps(self, states: torch.Tensor) -> torch.Tensor:
        """Get log-probabilities of all possible actions.

        Args:
            states: [batch_size, state_dim]

        Returns:
            Log-probability matrix: [batch_size, action_num]
        """
        return torch.log(self.get_probs(states))

    @abstractmethod
    def get_actions_and_logps_exploration(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get actions and corresponding log-probabilities under exploration mode.

        Args:
            states: [batch_size, state_dim]

        Returns:
            Actions and log-P values, both with shape [batch_size, 1].
        """
        pass

    def get_actions_and_logps_exploitation(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get actions and corresponding log-probabilities under exploitation mode.

        Args:
            states: [batch_size, state_dim]

        Returns:
            Actions and log-P values, both with shape [batch_size, 1].
        """
        action_prob = self.get_logps(states)  # (batch_size, num_actions)
        logps, action = action_prob.max(dim=1)
        return action, logps

    @abstractmethod
    def action_num(self) -> int:
        pass
