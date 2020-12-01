# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from components.agent_manager import DQNAgentManager, create_dqn_agents
from components.config import set_input_dim

from maro.rl import (
    AgentManagerMode, Component, Scheduler, TwoPhaseLinearExplorationParameterGenerator, concat_experiences_by_agent
)
from maro.simulator import Env
from maro.utils import Logger, convert_dottable


def launch(config, distributed_config):
    set_input_dim(config)
    config = convert_dottable(config)
    distributed_config = convert_dottable(distributed_config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    agent_manager = DQNAgentManager(
        name="distributed_cim_learner",
        mode=AgentManagerMode.TRAIN,
        agent_dict=create_dqn_agents(agent_id_list, config.agents)
    )

    scheduler = Scheduler(
        config.main_loop.max_episode,
        warmup_ep=config.main_loop.early_stopping.warmup_ep,
        exploration_parameter_generator_cls=TwoPhaseLinearExplorationParameterGenerator,
        exploration_parameter_generator_config=config.main_loop.exploration,
        logger=Logger("distributed_cim_learner", auto_timestamp=False)
    )

    if distributed_config.mode == "simple":
        from maro.rl import SimpleDistLearner as learner_cls
    elif distributed_config.mode == "seed":
        from maro.rl import SEEDLearner as learner_cls
    else:
        raise ValueError('Only "simple" and "seed" distributed modes are supported')

    learner = learner_cls(
        agent_manager,
        scheduler,
        concat_experiences_by_agent,
        expected_peers={Component.ACTOR.value: int(os.environ.get("NUM_ACTORS", distributed_config.num_actors))},
        group_name=os.environ["GROUP"] if "GROUP" in os.environ else distributed_config.group,
        redis_address=(distributed_config.redis.hostname, distributed_config.redis.port),
        max_retries=15
    )
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
    learner.exit()


if __name__ == "__main__":
    from components.config import config, distributed_config
    launch(config=config, distributed_config=distributed_config)
