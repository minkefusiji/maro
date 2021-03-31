# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from os import getenv
from os.path import dirname, join, realpath

import numpy as np

from maro.rl import LinearParameterScheduler, OffPolicyLearner
from maro.simulator import Env
from maro.utils import set_seeds

from examples.supply_chain.env_wrapper import SCEnvWrapper

from examples.supply_chain.dqn.agent import get_sc_agents


# Single-threaded launcher
if __name__ == "__main__":
    DEFAULT_CONFIG_PATH = join(dirname(realpath(__file__)), "config.yml")
    with open(getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
        config = yaml.safe_load(config_file)
    set_seeds(1024)  # for reproducibility
    env = Env(**config["training"]["env"])
    agent = get_sc_agents(env.agent_idx_list, config["agent"])
    scheduler = LinearParameterScheduler(config["training"]["max_episode"], **config["training"]["exploration"])
    learner = OffPolicyLearner(
        SCEnvWrapper(env), agent, scheduler,
        replay_memory_size=config["training"]["replay_memory"]["size"],
        replay_memory_overwrite_type=config["training"]["replay_memory"]["overwrite_type"]    
    )
    learner.run()