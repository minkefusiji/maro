# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Dict, List, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from maro.rl.types import State

from .core_model import AbsCoreModel, OptimOption


class DiscreteACNet(AbsCoreModel):
    """Model container for the actor-critic architecture for finite and discrete action spaces.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer options for the components.
            If none, no optimizer will be created for the model which means the model is not trainable.
            If it is a OptimOption instance, a single optimizer will be created to jointly optimize all
            parameters of the model. If it is a dictionary of OptimOptions, the keys will be matched against
            the component names and optimizers created for them. Note that it is possible to freeze certain
            components while optimizing others by providing a subset of the keys in ``component``.
            Defaults toNone.
        device (str): Identifier for the torch device. The model instance will be moved to the specified
            device. If it is None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise.
            Defaults to None.
    """
    def __init__(
        self,
        component: Union[nn.Module, Dict[str, nn.Module]],
        optim_option: Union[OptimOption, Dict[str, OptimOption]] = None,
        device: str = None
    ):
        super().__init__(component, optim_option=optim_option, device=device)

    @abstractmethod
    def forward(self, states: List[State], actor: bool = True, critic: bool = True) -> tuple:
        """Compute action probabilities and values for a state batch.

        The output is a tuple of (action_probs, values), where action probs is a tensor of shape
        (batch_size, action_space_size) and values is a tensor of shape (batch_size,). If only one
        of these two is needed, the return value for the other one can be set to None.

        Args:
            states (List[State]): State batch to compute action probabilities and values for.
            actor (bool): If True, the first element of the output will be actin probabilities. Defaults to True.
            critic (bool): If True, the second element of the output will be state values. Defaults to True.
        """
        raise NotImplementedError

    def get_action(self, states, max_prob: bool = False):
        """
        Given Q-values for a batch of states, return the action index and the corresponding maximum Q-value
        for each state.
        """
        action_probs, values = self.forward(states)
        if max_prob:
            probs, actions = action_probs.max(dim=1)
            return actions, torch.log(probs), values
        else:
            action_probs = Categorical(action_probs)  # (batch_size, action_space_size)
            actions = action_probs.sample()
            logps = action_probs.log_prob(actions)
            return actions, logps, values


class DiscretePolicyNet(AbsCoreModel):
    """Parameterized policy for finite and discrete action spaces.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer options for the components.
            If none, no optimizer will be created for the model which means the model is not trainable.
            If it is a OptimOption instance, a single optimizer will be created to jointly optimize all
            parameters of the model. If it is a dictionary of OptimOptions, the keys will be matched against
            the component names and optimizers created for them. Note that it is possible to freeze certain
            components while optimizing others by providing a subset of the keys in ``component``.
            Defaults toNone.
        device (str): Identifier for the torch device. The model instance will be moved to the specified
            device. If it is None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise.
            Defaults to None.
    """
    def __init__(
        self,
        component: Union[nn.Module, Dict[str, nn.Module]],
        optim_option: Union[OptimOption, Dict[str, OptimOption]] = None,
        device: str = None
    ):
        super().__init__(component, optim_option=optim_option, device=device)

    @abstractmethod
    def forward(self, states) -> torch.tensor:
        """Compute action probabilities corresponding to each state in ``states``.

        The output must be a torch tensor with shape (batch_size, action_space_size).

        Args:
            states (List[State]): State batch to compute action probabilities for.
        """
        raise NotImplementedError

    def get_action(self, states: List[State], max_prob: bool = False):
        """
        Given a batch of states, return actions selected based on the probabilities computed by ``forward``
        and the corresponding log probabilities.
        """
        action_prob = self.forward(states)   # (batch_size, num_actions)
        if max_prob:
            prob, action = action_prob.max(dim=1)
            return action, torch.log(prob)
        else:
            action_prob = Categorical(action_prob)  # (batch_size, action_space_size)
            action = action_prob.sample()
            log_p = action_prob.log_prob(action)
            return action, log_p


class DiscreteQNet(AbsCoreModel):
    """Q-value model for finite and discrete action spaces.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer options for the components.
            If none, no optimizer will be created for the model which means the model is not trainable.
            If it is a OptimOption instance, a single optimizer will be created to jointly optimize all
            parameters of the model. If it is a dictionary of OptimOptions, the keys will be matched against
            the component names and optimizers created for them. Note that it is possible to freeze certain
            components while optimizing others by providing a subset of the keys in ``component``.
            Defaults toNone.
        device (str): Identifier for the torch device. The model instance will be moved to the specified
            device. If it is None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise.
            Defaults to None.
    """
    def __init__(
        self,
        component: Union[nn.Module, Dict[str, nn.Module]],
        optim_option: Union[OptimOption, Dict[str, OptimOption]] = None,
        device: str = None
    ):
        super().__init__(component, optim_option=optim_option, device=device)

    @property
    @abstractmethod
    def num_actions(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, states: List[State]) -> torch.tensor:
        """Compute the Q-values for all actions as a tensor of shape (batch_size, action_space_size)."""
        raise NotImplementedError

    def get_action(self, states: List[State]):
        """
        Given Q-values for a batch of states and all actions, return the action index and the corresponding
        Q-values for each state.
        """
        q_for_all_actions = self.forward(states)  # (batch_size, num_actions)
        greedy_q, actions = q_for_all_actions.max(dim=1)
        return actions.detach(), greedy_q.detach(), q_for_all_actions.shape[1]

    def q_values(self, states, actions: torch.tensor):
        """Return the Q-values for a batch of states and actions."""
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(dim=1)
        q_for_all_actions = self.forward(states)  # (batch_size, num_actions)
        return q_for_all_actions.gather(1, actions).squeeze(dim=1)

    def soft_update(self, other_model: nn.Module, tau: float):
        """Soft-update model parameters using another model.

        Update formulae: param = (1 - tau) * param + tau * other_param.

        Args:
            other_model: The model to update the current model with.
            tau (float): Soft-update coefficient.
        """
        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data


class ContinuousACNet(AbsCoreModel):
    """Model container for the actor-critic architecture for continuous action spaces.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer options for the components.
            If none, no optimizer will be created for the model which means the model is not trainable.
            If it is a OptimOption instance, a single optimizer will be created to jointly optimize all
            parameters of the model. If it is a dictionary of OptimOptions, the keys will be matched against
            the component names and optimizers created for them. Note that it is possible to freeze certain
            components while optimizing others by providing a subset of the keys in ``component``.
            Defaults toNone.
        device (str): Identifier for the torch device. The model instance will be moved to the specified
            device. If it is None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise.
            Defaults to None.
    """
    def __init__(
        self,
        component: Union[nn.Module, Dict[str, nn.Module]],
        optim_option: Union[OptimOption, Dict[str, OptimOption]] = None,
        device: str = None
    ):
        super().__init__(component, optim_option=optim_option, device=device)

    def set_action_space(
        self,
        min_action: Union[float, np.ndarray] = None,
        max_action: Union[float, np.ndarray] = None
    ):
        """Set action clamping bounds.

        Args:
            min_action (Union[float, np.ndarray]): Lower bound for action. Actions generated from the model will be
                clipped according to this bound. Defaults to None, which means no lower bound.
            max_action (Union[float, np.ndarray]): Upper bound for action. Actions generated from the model will be
                clipped according to this bound. Defaults to None, which means no upper bound.
        """
        if min_action:
            assert isinstance(min_action, (float, np.ndarray)), "min_action must be a float or a numpy array"
        if max_action:
            assert isinstance(max_action, (float, np.ndarray)), "max_action must be a float or a numpy array"

        if isinstance(min_action, np.ndarray) and isinstance(max_action, np.ndarray):
            assert len(min_action) == len(max_action), "min_action and max_action should have the same dimension."

        # For torch clamping
        self._min_action = torch.from_numpy(min_action) if isinstance(min_action, np.ndarray) else min_action
        self._max_action = torch.from_numpy(max_action) if isinstance(max_action, np.ndarray) else max_action

    @abstractmethod
    def forward(self, states: List[State], actions=None) -> torch.tensor:
        """Compute actions for a batch of states or Q-values for a batch of states and actions.

        Args:
            states (List[State]): State batch to compute the Q-values for.
            actions: Action batch. If None, the output should be a batch of actions corresponding to
                the state batch. Otherwise, the output should be the Q-values for the given states and
                actions. Defaults to None.
        """
        raise NotImplementedError

    def get_action(self, states: List[State]) -> torch.tensor:
        """Compute actions given a batch of states."""
        return torch.clamp(self.forward(states), min=self._min_action, max=self._max_action)

    def value(self, states: List[State]):
        """Compute the Q-values for a batch of states using the actions computed from them."""
        return self.forward(states, actions=self.get_action(states))

    def soft_update(self, other_model: nn.Module, tau: float):
        """Soft-update model parameters using another model.

        Update formulae: param = (1 - tau) * param + tau * other_param.

        Args:
            other_model: The model to update the current model with.
            tau (float): Soft-update coefficient.
        """
        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data