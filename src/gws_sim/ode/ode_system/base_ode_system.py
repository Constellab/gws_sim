# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any
from abc import abstractmethod
from gws_core import ResourceSet, resource_decorator
from ..sim_system.sim_system import SimSystem


@resource_decorator("BaseODESystem", hide=True)
class BaseODESystem(ResourceSet):

    @abstractmethod
    def create_sim_system(self) -> SimSystem:
        """ Creates a SimSystem """

    @abstractmethod
    def dumps(self) -> Any:
        """ Dump the system to a destination data """

    @classmethod
    @abstractmethod
    def loads(cls, data: Any):
        """ Load from a source data """
