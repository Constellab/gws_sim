# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from abc import abstractmethod


class GeneralizedSimSystem:

    @abstractmethod
    def simulate(self, args=None) -> np.ndarray:
        """ Simulate the system """
