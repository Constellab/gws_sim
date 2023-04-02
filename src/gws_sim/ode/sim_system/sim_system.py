# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod

import numpy as np
from scipy.integrate import solve_ivp

from gws_core import BadRequestException


class SimSystem:

    def before_simulate(self):
        """
            Called before simulate
            To override if required
        """

    @abstractmethod
    def initial_state(self, args=None) -> np.ndarray:
        """ The initial tate of the system """

    @abstractmethod
    def parameters(self, t, args=None) -> np.ndarray:
        """ The derivative of the system """

    @abstractmethod
    def derivative(self, t: np.ndarray, x: np.ndarray, args=None) -> np.ndarray:
        """ The derivative of the system """

    def simulate(self, t_start, t_end, t_step=None, t_eval=None, method="RK45", args=None) -> np.ndarray:
        """ Simulate the system """
        if t_end <= t_start:
            raise BadRequestException("The final time must be greater than the initial time")

        self.before_simulate()
        if t_eval is None:
            if t_step is None:
                raise BadRequestException("Either argument `t_eval` or `t_step` is required")
            npoints = int((t_end-t_start) / t_step) + 1
            t_eval = np.linspace(t_start, t_end, num=npoints)

        solution = solve_ivp(
            fun=self.derivative,
            t_span=[t_start, t_end],
            y0=self.initial_state(args),
            method=method,
            t_eval=t_eval,
            args=args
        )

        return solution
