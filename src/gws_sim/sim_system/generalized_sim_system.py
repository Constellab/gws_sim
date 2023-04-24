# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from abc import abstractmethod


class GeneralizedSimSystem:

    @abstractmethod
    def simulate(
            self, t_start, t_end, initial_state=None, parameters=None, t_step=None, t_eval=None, method=None,
            args=None) -> np.ndarray:
        """ Simulate the system """

    @abstractmethod
    def simulate_with_cache(self, initial_state=None, parameters=None):
        """
        Simulate the system with cache data and new `initial_state`, `parameters`
        """
