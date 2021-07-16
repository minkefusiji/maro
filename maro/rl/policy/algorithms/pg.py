# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Tuple

import numpy as np
import torch

from maro.rl.experience import ExperienceStore, UniformSampler
from maro.rl.model import DiscretePolicyNet
from maro.rl.policy import AbsCorePolicy
from maro.rl.utils import get_truncated_cumulative_reward


class PolicyGradientConfig:
    """Configuration for the Policy Gradient algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        clear_experience_memory_every (int): Number of ``ActorCritic.learn`` calls between experience memory clearances.
            Defaults to 1.
    """
    __slots__ = ["reward_discount", "clear_experience_memory_every"]

    def __init__(self, reward_discount: float, clear_experience_memory_every: int = 1):
        self.reward_discount = reward_discount
        self.clear_experience_memory_every = clear_experience_memory_every


class PolicyGradient(AbsCorePolicy):
    """The vanilla Policy Gradient (VPG) algorithm, a.k.a., REINFORCE.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        policy_net (DiscretePolicyNet): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config (PolicyGradientConfig): Configuration for the PG algorithm.
        experience_store (ExperienceStore): An ``ExperienceStore`` instance for storing and retrieving experiences
            generated by the policy.
        experience_sampler_cls: Type of experience sampler. Must be a subclass of ``AbsSampler``. Defaults to
            ``UnifromSampler``.
        experience_sampler_kwargs (dict): Keyword arguments for ``experience_sampler_cls``.
        post_step (Callable): Custom function to be called after each gradient step. This can be used for tracking
            the learning progress. The function should have signature (loss, tracker) -> None. Defaults to None.
    """
    def __init__(
        self,
        policy_net: DiscretePolicyNet,
        config: PolicyGradientConfig,
        experience_store: ExperienceStore,
        experience_sampler_cls=UniformSampler,
        experience_sampler_kwargs: dict = {},
        post_step: Callable = None
    ):
        if not isinstance(policy_net, DiscretePolicyNet):
            raise TypeError("model must be an instance of 'DiscretePolicyNet'")
        super().__init__(
            experience_store,
            experience_sampler_cls=experience_sampler_cls,
            experience_sampler_kwargs=experience_sampler_kwargs
        )
        self.policy_net = policy_net
        self.config = config
        self._post_step = post_step
        self._num_learn_calls = 0
        self.device = self.policy_net.device

    def choose_action(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return actions and log probabilities for given states."""
        with torch.no_grad():
            actions, log_p = self.policy_net.get_action(states)
        actions, log_p = actions.cpu().numpy(), log_p.cpu().numpy()
        return (actions[0], log_p[0]) if len(actions) == 1 else actions, log_p

    def learn(self):
        """
        This should be called at the end of a simulation episode and the experiences obtained from
        the experience store's ``get`` method should be a sequential set, i.e., in the order in
        which they are generated during the simulation. Otherwise, the return values may be meaningless.
        """
        self.policy_net.train()
        experience_set = self.sampler.get()
        log_p = torch.from_numpy(np.asarray([act[1] for act in experience_set.actions])).to(self.device)
        rewards = torch.from_numpy(np.asarray(experience_set.rewards)).to(self.device)
        returns = get_truncated_cumulative_reward(rewards, self.config.reward_discount)
        returns = torch.from_numpy(returns).to(self.device)
        loss = -(log_p * returns).mean()
        self.policy_net.step(loss)

        if self._post_step:
            self._post_step(loss.detach().cpu().numpy(), self.tracker)

        self._num_learn_calls += 1
        if self._num_learn_calls % self.config.clear_experience_memory_every == 0:
            self.experience_store.clear()

    def set_state(self, policy_state):
        self.policy_net.load_state_dict(policy_state)

    def get_state(self):
        return self.policy_net.state_dict()
