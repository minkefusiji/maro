# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from abc import ABC, abstractmethod

from maro.communication import Proxy
from maro.communication.registry_table import RegisterTable
from maro.simulator import Env

from .common import Component, MessageTag


class AbsActor(ABC):
    """Abstract actor class."""
    def __init__(self, env: Env, **proxy_params):
        self._env = env
        self._proxy = Proxy(component_type=Component.ACTOR.value, **proxy_params)
        self._registry_table = RegisterTable(self._proxy.peers_name)
        self._registry_table.register_event_handler(f"*:{MessageTag.EXIT}:1", self.exit)

    @abstractmethod
    def roll_out(self, *args, **kwargs):
        raise NotImplementedError

    def launch(self):
        """Entry point method.

        This enters the actor into an infinite loop of listening to requests and handling them according to the
        register table. In this case, the only type of requests the actor needs to handle is roll-out requests.
        """
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            triggered_events = self._registry_table.get()
            for handler_fn, cached_messages in triggered_events:
                handler_fn(cached_messages)

    def exit(self):
        sys.exit(0)
