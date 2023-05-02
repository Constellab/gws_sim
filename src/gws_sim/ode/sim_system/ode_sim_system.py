# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod
from typing import List, Union

import numpy as np
from scipy.integrate import solve_ivp, odeint, OdeSolution

from gws_core import BadRequestException
from gws_sim.sim_system.generalized_sim_system import GeneralizedSimSystem


class ODESimSolution:
    t = None
    y = None
    success = None
    message = None

    def __init__(self, t, y, success, message):
        self.t = t
        self.y = y
        self.success = success
        self.message = message


class ODESimSystem(GeneralizedSimSystem):

    _cache: dict = None
    method = "ODEINT_ENGINE"

    def before_simulate(self, args):
        """
            Called before simulate
            To override if required
        """

    @abstractmethod
    def initial_state(self, args=None) -> np.ndarray:
        """ The initial state of the system """

    @abstractmethod
    def parameters(self, t, args=None) -> np.ndarray:
        """ The derivative of the system """

    @abstractmethod
    def state_names(self) -> List[str]:
        """ The state names """

    @abstractmethod
    def derivative(self, t: np.ndarray, x: np.ndarray, args=None) -> np.ndarray:
        """ The derivative of the system """

    def derivative_odeint(self, x: np.ndarray, t: np.ndarray, args=None) -> np.ndarray:
        """ The derivative of the system """
        return self.derivative(t, x, args)

    def derivative_ivp(self, t: np.ndarray, x: np.ndarray, args=None) -> np.ndarray:
        """ The derivative of the system """
        return self.derivative(t, x, args)

    def simulate(
            self, t_start, t_end, initial_state=None, parameters=None, t_step=None, t_eval=None, method=None,
            args=None) -> Union[OdeSolution, np.ndarray]:
        """ Simulate the system

        * Return OdeSolution by default (use `solve_ivp` engine)
        * Return np.ndarray if `ODEINT_ENGINE` is used as method
        """
        if t_end <= t_start:
            raise BadRequestException("The final time must be greater than the initial time")

        self.before_simulate(args)
        if t_eval is None:
            if t_step is None:
                raise BadRequestException("Either argument `t_eval` or `t_step` is required")
            npoints = int((t_end-t_start) / t_step) + 1
            t_eval = np.linspace(t_start, t_end, num=npoints)

        if parameters is None:
            parameters = self.parameters(t=t_start, args=args)
        if args is None:
            args = parameters
        else:
            args = [parameters, args]

        if initial_state is None:
            initial_state = self.initial_state(args)

        if method is None:
            method = self.method

        # create cached data
        self._cache = {
            "y0": initial_state,
            "t_eval": t_eval,
            "method": method,
            "t_start": t_start,
            "t_end": t_end,
        }

        if method == "ODEINT_ENGINE":
            solution = odeint(
                func=self.derivative_odeint,
                y0=initial_state,
                t=t_eval,
                args=(args,)
            )
            return ODESimSolution(t_eval, solution, True, "")
        else:
            solution = solve_ivp(
                fun=self.derivative_ivp,
                t_span=[t_start, t_end],
                y0=initial_state,
                method=method,
                t_eval=t_eval,
                args=(args,)
            )
            return ODESimSolution(solution.t, solution.y.T, solution.success, solution.message)

    def simulate_with_cache():
        pass
