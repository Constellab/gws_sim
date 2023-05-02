from typing import Tuple
from pandas import DataFrame
import pymc as pm
import numpy as np
from gws_core import BaseTestCase, Settings
from gws_sim import MCSampler

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


class LVSampler(MCSampler):

    def load_data(self):
        data = {
            "Time": list(np.arange(1900., 1921., 1)),
            "Lynx": [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                     8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6],
            "Hare": [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                     27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
        }
        return DataFrame(data=data)

    def get_observed_data(self):
        """ Returns the observed data """
        return self.get_data().iloc[:, 1:]

    def get_parameter_priors(self, args=None):
        """ parameters """
        alpha_h = 0.48
        beta_h = 0.02
        gamma_h = 0.93
        delta_h = 0.03
        xt0_h = 34.91
        yt0_h = 3.86

        alpha = pm.TruncatedNormal("alpha", mu=alpha_h, sigma=0.1, lower=0, initval=alpha_h)
        beta = pm.TruncatedNormal("beta", mu=beta_h, sigma=0.01, lower=0, initval=beta_h)
        gamma = pm.TruncatedNormal("gamma", mu=gamma_h, sigma=0.1, lower=0, initval=gamma_h)
        delta = pm.TruncatedNormal("delta", mu=delta_h, sigma=0.01, lower=0, initval=delta_h)
        xt0 = pm.TruncatedNormal("xto", mu=xt0_h, sigma=1, lower=0, initval=xt0_h)
        yt0 = pm.TruncatedNormal("yto", mu=yt0_h, sigma=1, lower=0, initval=yt0_h)
        sigma = pm.HalfNormal("sigma", 10)

        mu = (alpha, beta, gamma, delta, xt0, yt0)
        return mu, sigma

    def get_mu_estimate(self, mu: Tuple[pm.Distribution]):
        alpha, beta, gamma, delta, xt0, yt0 = mu
        params = (alpha, beta, gamma, delta,)
        # sol = odeint(
        #     func=lokta_voltera_odeint,
        #     y0=[xt0, yt0],
        #     t=self.get_data().loc[:, "Time"],
        #     args=(params,)
        # )
        # return sol

        sol = solve_ivp(
            fun=lokta_voltera_solve_ivp,
            t_span=[self.get_data().iat[0, 0], self.get_data().iat[-1, 0]],
            t_eval=self.get_data().loc[:, "Time"].values.tolist(),
            y0=[xt0, yt0],
            vectorized=True,
            method="LSODA",
            args=(params,)
        )
        return sol.y.T

    def get_data_likelihood_type(self):
        return pm.Normal


class TestMCSampler(BaseTestCase):

    def test_nonlin(self):
        sampler = LVSampler()

        # Test if data are not empty
        data = sampler.get_data()
        self.assertIsNotNone(data)
        # Test if data is equal to the data we loaded
        data_loaded = DataFrame({
            "Time": list(np.arange(1900., 1921., 1)),
            "Lynx": [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                     8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6],
            "Hare": [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                     27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
        })
        self.assertTrue(data.equals(data_loaded))

        sampler.sample()

        # # Test parameter_priors
        # with pm.Model() as model:
        #     mu, sigma = sampler.get_parameter_priors()
        # self.assertIsNotNone(mu)
        # self.assertIsNotNone(sigma)
        # self.assertIsInstance(mu, tuple)
        # # Sigma n'est pas une instance de pm.Distribution mais un TensorVariable car Distribution le retourne comme ça
        # self.assertIsInstance(sigma, TensorVariable)

        # # Test mu estimate
        # mu_estimate = sampler.get_mu_estimate(mu)
        # self.assertIsNotNone(mu_estimate)
        # self.assertIsInstance(mu_estimate, tuple)

        # # Test data likelihood type
        # likelihood_type = sampler.get_data_likelihood_type()
        # self.assertIsNotNone(likelihood_type)
        # self.assertEqual(likelihood_type, pm.Normal)

        # # Test sample
        # # TODO : ne fonctionne pas
        # # trace = sampler.sample(args = (mu[0], mu[1], mu[2], mu[3], mu[4], mu[5]))
        # # self.assertIsNotNone(trace)

        # # print(az.summary(trace))
