import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from scipy.integrate import odeint
from numba import njit

from abc import abstractmethod
from typing import List

import numpy as np
from scipy.integrate import solve_ivp

from gws_core import BadRequestException
from .ode_sim_system import ODESimSystem


class MCMCSystem(ODESimSystem):

    def before_simulate(self):
        """
            Called before simulate
            To override if required
        """
        a = 5
        return a

    def initial_state(self, args=None) -> np.ndarray:
        """ The initial state of the system """

    def parameters(self, t, args=None) -> np.ndarray:
        """ The derivative of the system """

    def state_names(self) -> List[str]:
        """ The state names """

    def derivative(self, t: np.ndarray, x: np.ndarray, args=None) -> np.ndarray:
        """ The derivative of the system """

    @njit
    def rhs(self, X, t, theta):
        # unpack parameters
        x, y = X
        alpha, beta, gamma, delta, xt0, yt0 = theta
        # equations
        dx_dt = alpha * x - beta * x * y
        dy_dt = -gamma * y + delta * x * y
        return [dx_dt, dy_dt]

    #TODO : changer la fonction odeint
    # decorator with input and output types a Pytensor double float tensors
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(self, theta, data):
        #sim_system = self.sim_system
        names = data.column_names
        #result_py = odeint(func=sim_system.derivative, y0=theta[-2:], t=data.get_column_data(names[0]), args=(theta,))
        result_py = odeint(func=self.rhs, y0=[11.85718184,  5.9927679 ], t=data.get_column_data(names[0]), args=(theta,))#
        return result_py
