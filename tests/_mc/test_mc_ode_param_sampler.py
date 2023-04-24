
from gws_core import BaseTestCase, Settings
from gws_sim import MCODEParamSampler, ODESimSystem
import pymc as pm
import numpy as np
from pandas import DataFrame
import arviz as az
from numba import njit

settings = Settings.get_instance()


class LoktaVoltera(ODESimSystem):

    method = "ODEINT_ENGINE"

    def state_names(self):
        """ state_names """
        return "Lynx", "Hare"

    def initial_state(self, args=None):
        """ initial_state """
        return (4.0, 30.0)

    def parameters(self, t, args=None):
        """ parameters """
        # args = (alpha, beta, gamma, delta, x0, y0)
        # return the first 4 values
        return args[0:4]

    @staticmethod
    @njit
    def optimized_derivative(t, x, params):
        """ derivative """
        X, Y = x
        alpha, beta, gamma, delta = params
        dX_dt = alpha * X - beta * X * Y
        dY_dt = -gamma * Y + delta * X * Y
        return [dX_dt, dY_dt]

    def derivative(self, t, x, args=None):
        """ derivative """
        params = self.parameters(t, args)
        return self.optimized_derivative(t, x, params)


class LVEstimator(MCODEParamSampler):

    def load_data(self):
        data = {
            "Time": list(np.arange(1900., 1921., 1)),
            "Lynx": [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                     8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6],
            "Hare": [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                     27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
        }
        return DataFrame(data=data)

    def create_sim_system(self):
        """ The ODE System """
        return LoktaVoltera()

    def get_parameter_priors(self, args=None):
        """ parameters """
        alpha_h = 0.48
        beta_h = 0.02
        gamma_h = 0.93
        delta_h = 0.03
        h0_h = 34.91
        l0_h = 3.86

        alpha = pm.TruncatedNormal("alpha", mu=alpha_h, sigma=0.1, lower=0, initval=alpha_h)
        beta = pm.TruncatedNormal("beta", mu=beta_h, sigma=0.01, lower=0, initval=beta_h)
        gamma = pm.TruncatedNormal("gamma", mu=gamma_h, sigma=0.1, lower=0, initval=gamma_h)
        delta = pm.TruncatedNormal("delta", mu=delta_h, sigma=0.01, lower=0, initval=delta_h)
        xt0 = pm.TruncatedNormal("xto", mu=h0_h, sigma=1, lower=0, initval=h0_h)
        yt0 = pm.TruncatedNormal("yto", mu=l0_h, sigma=1, lower=0, initval=l0_h)
        sigma = pm.HalfNormal("sigma", 10)

        mu = (alpha, beta, gamma, delta, xt0, yt0)
        return mu, sigma

    def get_data_likelihood_type(self):
        return pm.Normal


class TestMCSim(BaseTestCase):

    def test_nonlin(self):
        estimator = LVEstimator()
        result = estimator.sample()

        traces = result.get_traces()
        print(az.summary(traces))

        print(result.get_param_traces())
