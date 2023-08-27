
import numpy as np
from gws_sim import MCSystemHelper

from scipy.integrate import odeint
from numba import njit
from pandas import DataFrame
from typing import Tuple
import pymc as pm


@njit
def lokta_voltera_odeint(x, t, params):
    """ derivative """
    X, Y = x
    alpha, beta, gamma, delta = params
    dXdt = alpha * X - beta * X * Y
    dYdt = -gamma * Y + delta * X * Y
    return [dXdt, dYdt]


class System(MCSystemHelper):

    def load_data(self):
        return None

    def load_priors(self):
        return None

    def get_mu_estimate(self, mu: Tuple[pm.Distribution]):
        alpha, beta, gamma, delta, xt0, yt0 = mu
        params = (alpha, beta, gamma, delta,)
        sol = odeint(
            func=lokta_voltera_odeint,
            y0=[xt0, yt0],
            t=self.get_data().loc[:, "Time"],
            args=(params,)
        )
        return sol

        # sol = solve_ivp(
        #     fun=lokta_voltera_solve_ivp,
        #     t_span=[self.get_data().iat[0, 0], self.get_data().iat[-1, 0]],
        #     t_eval=self.get_data().loc[:, "Time"].values.tolist(),
        #     y0=[xt0, yt0],
        #     vectorized=True,
        #     method="LSODA",
        #     args=(params,)
        # )
        # return sol.y.T
