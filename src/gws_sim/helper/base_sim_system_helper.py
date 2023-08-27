# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod
from typing import Union

import numpy as np


class BaseSimSystemHelper:

    @abstractmethod
    def simulate(self, *args, **kwargs) -> Union[np.ndarray]:
        pass
