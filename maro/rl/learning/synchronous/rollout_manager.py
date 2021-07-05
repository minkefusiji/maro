# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Pipe, Process
from os import getcwd
from random import choices
from typing import Callable

from maro.communication import Proxy, SessionType
from maro.rl.experience import ExperienceSet
from maro.rl.utils import MsgKey, MsgTag
from maro.utils import Logger

from ..agent_wrapper import AgentWrapper
from ..env_wrapper import AbsEnvWrapper
from .rollout_worker import rollout_worker_process


class AbsRolloutManager(ABC):
    """Controller for simulation data collection."""
    def __init__(self):
        super().__init__()
        self.episode_complete = False

    @abstractmethod
    def collect(self, ep: int, segment: int, policy_state_dict: dict, version: int):
        """Collect simulation data, i.e., experiences for training.

        Args:
            ep (int): Current episode index.
            segment (int): Current segment index.
            policy_state_dict (dict): Policy states to use for simulation.
            version (int): Version index from the policy manager from which the ``policy_state_dict`` is obtained.

        Returns:
            Experiences for policy training.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, ep: int, policy_state_dict: dict):
        """Evaluate the performance of ``policy_state_dict``.

        Args:
            ep (int): Current training episode index.
            policy_state_dict (dict): Policy states to use for simulation.

        Returns:
            Environment summary.
        """
        raise NotImplementedError

    def reset(self):
        self.episode_complete = False


class LocalRolloutManager(AbsRolloutManager):
    """Local roll-out controller.

    Args:
        env_wrapper (AbsEnvWrapper): An ``AbsEnvWrapper`` instance to interact with a set of agents and collect
            experiences for policy training / update.
        agent_wrapper (AgentWrapper): Agent wrapper to interact with the environment wrapper.
        num_steps (int): Number of environment steps to roll out in each call to ``collect``. Defaults to -1, in which
            case the roll-out will be executed until the end of the environment.
        eval_env_wrapper (AbsEnvWrapper): An ``AbsEnvWrapper`` instance for policy evaluation. If None, ``env`` will be
            used as the evaluation environment. Defaults to None.
        log_env_summary (bool): If True, the ``summary`` property of the environment wrapper will be logged at the end
            of each episode. Defaults to True.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "LOCAL_ROLLOUT_MANAGER" will be created at
            init time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """

    def __init__(
        self,
        env_wrapper: AbsEnvWrapper,
        agent_wrapper: AgentWrapper,
        num_steps: int = -1,
        eval_env_wrapper: AbsEnvWrapper = None,
        log_env_summary: bool = True,
        log_dir: str = getcwd(),
    ):
        if num_steps == 0 or num_steps < -1:
            raise ValueError("num_steps must be a positive integer or -1")

        super().__init__()
        self._logger = Logger("LOCAL_ROLLOUT_MANAGER", dump_folder=log_dir)

        self.env = env_wrapper
        self.eval_env = eval_env_wrapper if eval_env_wrapper else self.env
        self.agent = agent_wrapper

        self._num_steps = num_steps if num_steps > 0 else float("inf")
        self._log_env_summary = log_env_summary

    def collect(self, ep: int, segment: int, policy_state_dict: dict, version: int):
        """Collect simulation data, i.e., experiences for training.

        Args:
            ep (int): Current episode index.
            segment (int): Current segment index.
            policy_state_dict (dict): Policy states to use for simulation.
            version (int): Version index from the policy manager from which the ``policy_state_dict`` is obtained.

        Returns:
            Experiences for policy training.
        """
        self._logger.info(f"Collecting simulation data (episode {ep}, segment {segment}, policy version {version})")
        t0 = time.time()
        learning_time = 0
        num_experiences_collected = 0

        # start of new episode
        if self.env.state is None:
            self.env.reset()
            self.env.start()  # get initial state
            self.agent.exploration_step()

        # set policy states
        self.agent.set_policy_states(policy_state_dict)
        # update exploration parameters
        self.agent.explore()

        start_step_index = self.env.step_index + 1
        steps_to_go = self._num_steps
        while self.env.state and steps_to_go > 0:
            action = self.agent.choose_action(self.env.state)
            self.env.step(action)
            steps_to_go -= 1

        self._logger.info(
            f"Roll-out finished for ep {ep}, segment {segment}"
            f"(steps {start_step_index} - {self.env.step_index})"
        )

        # update the exploration parameters if an episode is finished
        if not self.env.state:
            self.episode_complete = True
            # performance details
            if self._log_env_summary:
                self._logger.info(f"ep {ep}: {self.env.summary}")

            self._logger.debug(
                f"ep {ep} summary - "
                f"running time: {time.time() - t0} "
                f"env steps: {self.env.step_index} "
                f"learning time: {learning_time} "
                f"experiences collected: {num_experiences_collected}"
            )

        return self.env.get_experiences()

    def evaluate(self, ep: int, policy_state_dict: dict):
        """Evaluate the performance of ``policy_state_dict``.

        Args:
            ep (int): Current training episode index.
            policy_state_dict (dict): Policy states to use for simulation.

        Returns:
            Environment summary.
        """
        self._logger.info("Evaluating...")
        self.agent.set_policy_states(policy_state_dict)
        self.agent.exploit()
        self.eval_env.reset()
        self.eval_env.start()  # get initial state
        while self.eval_env.state:
            action = self.agent.choose_action(self.eval_env.state)
            self.eval_env.step(action)

        if self._log_env_summary:
            self._logger.info(f"Evaluation result: {self.eval_env.summary}")

        return self.eval_env.summary


class MultiProcessRolloutManager(AbsRolloutManager):
    """Roll-out manager that spawns a set of roll-out worker processes for parallel data collection.

    Args:
        num_workers (int): Number of remote roll-out workers.
        create_env_wrapper_func (Callable): Function to be used by each spawned roll-out worker to create an
            environment wrapper for training data collection. The function should take no parameters and return an
            environment wrapper instance.
        create_agent_wrapper_func (Callable): Function to be used by each spawned roll-out worker to create a
            decision generator for interacting with the environment. The function should take no parameters and return
            a ``AgentWrapper`` instance.
        create_env_wrapper_func (Callable): Function to be used by each spawned roll-out worker to create an
            environment wrapper for evaluation. The function should take no parameters and return an environment
            wrapper instance. If this is None, the training environment wrapper will be used for evaluation in the
            worker processes. Defaults to None.
        num_steps (int): Number of environment steps to roll out in each call to ``collect``. Defaults to -1, in which
            case the roll-out will be executed until the end of the environment.
        num_eval_workers (int): Number of workers for evaluation. Defaults to 1.
        log_env_summary (bool): If True, the ``summary`` property of the environment wrapper will be logged at the end
            of each episode. Defaults to True.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "LOCAL_ROLLOUT_MANAGER" will be created at
            init time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        num_workers: int,
        create_env_wrapper_func: Callable[[], AbsEnvWrapper],
        create_agent_wrapper_func: Callable[[], AgentWrapper],
        create_eval_env_wrapper_func: Callable[[], AbsEnvWrapper] = None,
        num_steps: int = -1,
        num_eval_workers: int = 1,
        log_env_summary: bool = True,
        log_dir: str = getcwd(),
    ):
        super().__init__()
        self._logger = Logger("ROLLOUT_MANAGER", dump_folder=log_dir)
        self._num_workers = num_workers
        self._num_steps = num_steps
        self._log_env_summary = log_env_summary
        self._num_eval_workers = num_eval_workers
        self.total_experiences_collected = 0
        self.total_env_steps = 0
        self._exploration_step = False

        self._worker_processes = []
        self._manager_ends = []
        for index in range(self._num_workers):
            manager_end, worker_end = Pipe()
            self._manager_ends.append(manager_end)
            worker = Process(
                target=rollout_worker_process,
                args=(
                    index,
                    worker_end,
                    create_env_wrapper_func,
                    create_agent_wrapper_func,
                ),
                kwargs={
                    "create_eval_env_wrapper_func": create_eval_env_wrapper_func,
                    "log_dir": log_dir
                }
            )
            self._worker_processes.append(worker)
            worker.start()

    def collect(self, ep: int, segment: int, policy_state_dict: dict, version: int):
        """Collect simulation data, i.e., experiences for training.

        Args:
            ep (int): Current episode index.
            segment (int): Current segment index.
            policy_state_dict (dict): Policy states to use for simulation.
            version (int): Version index from the policy manager from which the ``policy_state_dict`` is obtained.

        Returns:
            Experiences for policy training.
        """
        rollout_req = {
            "type": "collect",
            "episode": ep,
            "segment": segment,
            "num_steps": self._num_steps,
            "policy": policy_state_dict,
            "exploration_step": self._exploration_step
        }

        for conn in self._manager_ends:
            conn.send(rollout_req)

        self._logger.info(f"Collecting simulation data (episode {ep}, segment {segment}, policy version {version})")

        if self._exploration_step:
            self._exploration_step = False

        combined_exp_by_policy = defaultdict(ExperienceSet)
        for conn in self._manager_ends:
            result = conn.recv()
            exp_by_policy = result["experiences"]
            self.total_experiences_collected += sum(exp.size for exp in exp_by_policy.values())
            self.total_env_steps += result["num_steps"]

            for policy_name, exp in exp_by_policy.items():
                combined_exp_by_policy[policy_name].extend(exp)

            # log roll-out summary
            self.episode_complete = result["episode_end"]
            if self.episode_complete and self._log_env_summary:
                env_summary = result["env_summary"]
                self._logger.info(f"env summary: {env_summary}")

        if self.episode_complete:
            self._exploration_step = True

        return combined_exp_by_policy

    def evaluate(self, ep: int, policy_state_dict: dict):
        """Evaluate the performance of ``policy_state_dict``.

        Args:
            ep (int): Current training episode index.
            policy_state_dict (dict): Policy states to use for simulation.

        Returns:
            Environment summary.
        """
        eval_worker_conns = choices(self._manager_ends, k=self._num_eval_workers)
        for conn in eval_worker_conns:
            conn.send({"type": "evaluate", "episode": ep, "policy": policy_state_dict})

        env_summary_dict = {}
        for conn in self._manager_ends:
            result = conn.recv()
            env_summary_dict[result["worker_id"]] = result["env_summary"]

        return env_summary_dict

    def exit(self):
        """Tell the worker processes to exit."""
        for conn in self._manager_ends:
            conn.send({"type": "quit"})


class MultiNodeRolloutManager(AbsRolloutManager):
    """Controller for a set of remote roll-out workers, possibly distributed on different computation nodes.

    Args:
        group (str): Group name for the roll-out cluster, which includes all roll-out workers and a roll-out manager
            that manages them.
        num_workers (int): Number of remote roll-out workers.
        num_steps (int): Number of environment steps to roll out in each call to ``collect``. Defaults to -1, in which
            case the roll-out will be executed until the end of the environment.
        max_receive_attempts (int): Maximum number of attempts to receive  results in ``collect``. Defaults to
            None, in which case the number is set to ``num_workers``.
        receive_timeout (int): Maximum wait time (in milliseconds) for each attempt to receive from the workers. This
            This multiplied by ``max_receive_attempts`` give the upperbound for the amount of time to receive the
            desired amount of data from workers. Defaults to None, in which case each receive attempt is blocking.
        max_lag (int): Maximum policy version lag allowed for experiences collected from remote roll-out workers.
            Experiences collected using policy versions older than (current_version - max_lag) will be discarded.
            Defaults to 0, in which case only experiences collected using the latest policy version will be returned.
        num_eval_workers (int): Number of workers for evaluation. Defaults to 1.
        log_env_summary (bool): If True, the ``summary`` property of the environment wrapper will be logged at the end
            of each episode. Defaults to True.
        proxy_kwargs: Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to the empty dictionary.
        log_dir (str): Directory to store logs in. A ``Logger`` with tag "LOCAL_ROLLOUT_MANAGER" will be created at
            init time and this directory will be used to save the log files generated by it. Defaults to the current
            working directory.
    """
    def __init__(
        self,
        group: str,
        num_workers: int,
        num_steps: int = -1,
        max_receive_attempts: int = None,
        receive_timeout: int = None,
        max_lag: int = 0,
        num_eval_workers: int = 1,
        log_env_summary: bool = True,
        proxy_kwargs: dict = {},
        log_dir: str = getcwd()
    ):
        if num_eval_workers > num_workers:
            raise ValueError("num_eval_workers cannot exceed the number of available workers")

        super().__init__()
        self.num_workers = num_workers
        peers = {"rollout_worker": num_workers}
        self._proxy = Proxy(group, "rollout_manager", peers, component_name="ROLLOUT_MANAGER", **proxy_kwargs)
        self._workers = self._proxy.peers["rollout_worker"]  # remote roll-out worker ID's
        self._logger = Logger(self._proxy.name, dump_folder=log_dir)

        self._num_steps = num_steps

        if max_receive_attempts is None:
            max_receive_attempts = self.num_workers
            self._logger.info(f"Maximum receive attempts is set to {max_receive_attempts}")

        self.max_receive_attempts = max_receive_attempts
        self.receive_timeout = receive_timeout

        self._max_lag = max_lag
        self.total_experiences_collected = 0
        self.total_env_steps = 0
        self._log_env_summary = log_env_summary

        self._num_eval_workers = num_eval_workers

        self._exploration_step = False

    def collect(self, ep: int, segment: int, policy_state_dict: dict, version: int):
        """Collect simulation data, i.e., experiences for training.

        Args:
            ep (int): Current episode index.
            segment (int): Current segment index.
            policy_state_dict (dict): Policy states to use for simulation.
            version (int): Version index from the policy manager from which the ``policy_state_dict`` is obtained.

        Returns:
            Experiences for policy training.
        """
        msg_body = {
            MsgKey.EPISODE: ep,
            MsgKey.SEGMENT: segment,
            MsgKey.NUM_STEPS: self._num_steps,
            MsgKey.POLICY_STATE: policy_state_dict,
            MsgKey.VERSION: version,
            MsgKey.EXPLORATION_STEP: self._exploration_step
        }

        self._proxy.iscatter(MsgTag.COLLECT, SessionType.TASK, [(worker_id, msg_body) for worker_id in self._workers])
        self._logger.info(f"Collecting simulation data (episode {ep}, segment {segment}, policy version {version})")

        if self._exploration_step:
            self._exploration_step = False

        # Receive roll-out results from remote workers
        combined_exp_by_policy = defaultdict(ExperienceSet)
        num_finishes = 0
        for _ in range(self.max_receive_attempts):
            msg = self._proxy.receive_once(timeout=self.receive_timeout)
            if msg.tag != MsgTag.COLLECT_DONE:
                self._logger.info(
                    f"Ignored a message of type {msg.tag} (expected message type {MsgTag.COLLECT_DONE})"
                )
                continue

            if version - msg.body[MsgKey.VERSION] > self._max_lag:
                self._logger.info(
                    f"Ignored a message because it contains experiences generated using a stale policy version. "
                    f"Expected experiences generated using policy versions no earlier than {version - self._max_lag} "
                    f"got {msg.body[MsgKey.VERSION]}"
                )
                continue

            exp_by_policy = msg.body[MsgKey.EXPERIENCES]
            self.total_experiences_collected += sum(exp.size for exp in exp_by_policy.values())
            self.total_env_steps += msg.body[MsgKey.NUM_STEPS]

            for policy_name, exp in exp_by_policy.items():
                combined_exp_by_policy[policy_name].extend(exp)

            if msg.body[MsgKey.SEGMENT] == segment:
                self.episode_complete = msg.body[MsgKey.EPISODE_END]
                if self.episode_complete:
                    # log roll-out summary
                    if self._log_env_summary:
                        self._logger.info(f"env summary: {msg.body[MsgKey.ENV_SUMMARY]}")
                num_finishes += 1
                if num_finishes == self.num_workers:
                    break

        if self.episode_complete:
            self._exploration_step = True

        return combined_exp_by_policy

    def evaluate(self, ep: int, policy_state_dict: dict):
        """Evaluate the performance of ``policy_state_dict``.

        Args:
            ep (int): Current training episode index.
            policy_state_dict (dict): Policy states to use for simulation.

        Returns:
            Environment summary.
        """
        msg_body = {MsgKey.EPISODE: ep, MsgKey.POLICY_STATE: policy_state_dict}

        workers = choices(self._workers, k=self._num_eval_workers)
        env_summary_dict = {}
        self._proxy.iscatter(MsgTag.EVAL, SessionType.TASK, [(worker_id, msg_body) for worker_id in workers])
        self._logger.info(f"Sent evaluation requests to {workers}")

        # Receive roll-out results from remote workers
        num_finishes = 0
        for msg in self._proxy.receive():
            if msg.tag != MsgTag.EVAL_DONE or msg.body[MsgKey.EPISODE] != ep:
                self._logger.info(
                    f"Ignore a message of type {msg.tag} with episode index {msg.body[MsgKey.EPISODE]} "
                    f"(expected message type {MsgTag.EVAL_DONE} and episode index {ep})"
                )
                continue

            env_summary_dict[msg.source] = msg.body[MsgKey.ENV_SUMMARY]

            if msg.body[MsgKey.EPISODE] == ep:
                num_finishes += 1
                if num_finishes == self._num_eval_workers:
                    break

        return env_summary_dict

    def exit(self):
        """Tell the remote workers to exit."""
        self._proxy.ibroadcast("rollout_worker", MsgTag.EXIT, SessionType.NOTIFICATION)
        self._proxy.close()
        self._logger.info("Exiting...")
