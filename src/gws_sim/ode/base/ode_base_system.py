# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod
from typing import Any

import numpy as np
from gws_core import ResourceSet, resource_decorator


@resource_decorator("BaseODESystem", hide=True)
class BaseODESystem(ResourceSet):

    DEFAULT_INITIAL_STATE_TABLE_NAME = "Default unitial state"

    _initial_state: Any = None

    @abstractmethod
    def derivative(self, t: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """ The derivative of the ODE system """

    @property
    @abstractmethod
    def size(self) -> int:
        """ The size if the state vector of the ODE system """

    @property
    @abstractmethod
    def state_names(self):
        """ The names of the state variables """

    # -- G --

    @abstractmethod
    def get_initial_state(self, as_float=True):
        """ Get the initial state """

    @abstractmethod
    def get_default_initial_state(self):
        """ Get the default initial state """

    # -- S --

    @abstractmethod
    def set_initial_state(self, initial_state: Any):
        """ Set the initial state """

    @abstractmethod
    def set_default_initial_state(self, initial_state: Any):
        """ Set the default initial state """
