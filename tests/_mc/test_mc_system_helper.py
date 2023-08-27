from typing import Tuple
from pandas import DataFrame
import pymc as pm
import numpy as np
from gws_core import BaseTestCase, Settings
from gws_sim import MCSystemHelper
# import os

from pytensor.tensor.var import TensorVariable
from scipy.integrate import odeint, solve_ivp
from numba import njit

settings = Settings.get_instance()


@njit
def lokta_voltera_odeint(x, t, params):
    """ derivative """
    X, Y = x
    alpha, beta, gamma, delta = params
    dXdt = alpha * X - beta * X * Y
    dYdt = -gamma * Y + delta * X * Y
    return [dXdt, dYdt]


@njit
def lokta_voltera_solve_ivp(t, x, params):
    """ derivative """
    X, Y = x
    alpha, beta, gamma, delta = params
    dXdt = alpha * X - beta * X * Y
    dYdt = -gamma * Y + delta * X * Y
    return [dXdt, dYdt]


class LVSystem(MCSystemHelper):

    def load_data(self):
        data = {
            "Time": list(np.arange(1900., 1921., 1)),
            "Lynx": [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                     8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6],
            "Hare": [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                     27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
        }
        return DataFrame(data=data)

    def load_priors(self):
        """ parameters """
        alpha_h = 0.48
        beta_h = 0.02
        gamma_h = 0.93
        delta_h = 0.03
        xt0_h = 34.91
        yt0_h = 3.86

        mu = [
            {"name": "alpha", "type": "TruncatedNormal", "mu": alpha_h, "sigma": 0.1, "lower": 0, "initval": alpha_h},
            {"name": "beta", "type": "TruncatedNormal", "mu": beta_h, "sigma": 0.1, "lower": 0, "initval": beta_h},
            {"name": "gamma", "type": "TruncatedNormal", "mu": gamma_h, "sigma": 0.1, "lower": 0, "initval": gamma_h},
            {"name": "delta", "type": "TruncatedNormal", "mu": delta_h, "sigma": 0.1, "lower": 0, "initval": delta_h},
            {"name": "xto", "type": "TruncatedNormal", "mu": xt0_h, "sigma": 0.1, "lower": 0, "initval": xt0_h},
            {"name": "yto", "type": "TruncatedNormal", "mu": yt0_h, "sigma": 0.1, "lower": 0, "initval": yt0_h},
        ]

        likelihood = {
            "type": "Normal",
            "params": [{"name": "sigma", "type": "HalfNormal", "sigma": 10}]
        }

        return {"mu": mu, "likelihood": likelihood}

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

    # def get_data_likelihood_type(self):
    #     return pm.Normal


class TestMCSystemHelper(BaseTestCase):

    def test_nonlin(self):
        system = LVSystem()

        # Test if data are not empty
        data = system.get_data()
        self.assertIsNotNone(data)
        # Test if data is equal to the data we loaded
        data_loaded = DataFrame({
            "Time": list(np.arange(1900., 1921., 1)),
            "Lynx": [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                     8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6],
            "Hare": [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                     27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
        })

        # settings = Settings.get_instance()
        # data_dir = settings.get_variable("gws_sim:testdata_dir")
        # data_loaded.to_csv(os.path.join(data_dir, "lv_data.csv"), index=False)
        # return

        self.assertTrue(data.equals(data_loaded))

        result = system.sample()

        trace = result.get_parameter_traces()
        pred = result.get_state_predictions()
        pred_stats = result.get_state_prediction_stats()

        print(trace)
        print(pred)
        print(pred_stats)

        self.assertTrue('Lynx' in pred)
        self.assertTrue('Hare' in pred)
        self.assertEqual(pred_stats[0].shape, (21, 2))
        self.assertEqual(pred_stats[1].shape, (21, 2))
