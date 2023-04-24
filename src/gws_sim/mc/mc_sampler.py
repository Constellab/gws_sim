# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod
from typing import List, Tuple, Type

from pandas import DataFrame
import numpy as np
import pandas as pd

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

from gws_core import BadRequestException
from ..sim_system.generalized_sim_system import GeneralizedSimSystem


class MCSamplerResult:
    """ Linear regression data """
    _traces = None
    _sampler: 'MCSampler' = None

    def __init__(self, traces, sampler):
        self._traces = traces
        self._sampler = sampler

    def get_traces(self):
        return self._traces

    def get_param_traces(self, num_samples=None):
        dfs = []
        for colname in self._sampler.get_param_names():
            data = az.extract(self._traces, num_samples=num_samples, group='posterior',
                              combined=True, var_names=colname)
            dfs.append(DataFrame(data.T, columns=[colname]))
        return pd.concat(dfs, axis=1)

    def get_predictions(self, num_samples=None):
        param_traces = self.get_param_traces(num_samples=num_samples)
        preds = []

        for i in range(0, num_samples):
            mu = param_traces.iloc[i, :].values.tolist()
            sol = self._sampler.get_mu_estimate(mu)

            if preds is None:
                m, n = sol.y.shape
                for k in range(n):
                    preds[k] = np.zeros(shape=(m, num_samples))

            m, n = sol.y.shape
            for k in range(n):
                preds[k][:, i] = sol.y[:, k]

        for k in len(preds):
            pass

        return sol.t, preds

    def get_prediction_stats(self, num_samples=None):
        t, preds = self.get_predictions(num_samples=num_samples)
        pred_std = []
        pred_mean = []
        for pred in preds:
            pred_std = pred.std(axis=1)
            pred_mean = pred.mean(axis=1)
            pred_stats = pd.concat([pred_mean, pred_std], axis=1)
            pred_stats.columns = ["mean", "std"]
        return t, pred_stats


class MCSampler:
    _cached_data: DataFrame = None
    _cached_sim_system: GeneralizedSimSystem = None
    _param_names: List[str] = None

    # -- C --

    @abstractmethod
    def create_sim_system(self) -> Type[GeneralizedSimSystem]:
        """ Returns the sim system """

    def _create_model(self, args=None):

        @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
        def pytensor_forward_model_matrix(mu):
            return self.get_mu_estimate(mu)

        with pm.Model() as model:
            mu, sigma = self.get_parameter_priors(args=args)
            mu_hat = pytensor_forward_model_matrix(pm.math.stack(mu))
            likelihood = self.get_data_likelihood_type()
            likelihood(
                "y_obs",
                mu=mu_hat,
                sigma=sigma,
                observed=self.get_observed_data()
            )
            self._param_names = [*[str(p) for p in mu], str(sigma)]
            return model

    # -- G --

    def get_sim_system(self) -> Type[GeneralizedSimSystem]:
        """ Returns the sim system """
        if self._cached_sim_system is None:
            self._cached_sim_system = self.create_sim_system()
        return self._cached_sim_system

    def get_data(self):
        """ Returns the data """
        if self._cached_data is None:
            data = self.load_data()
            if data is None:
                raise BadRequestException("No data defined. Please set data or define the `load_data()` method")

            self._cached_data = data
        return self._cached_data

    def get_observed_data(self):
        """
        Returns the observed data

        To override if required
        """
        return self.get_data()

    def get_param_names(self) -> List[str]:
        return self._param_names

    @ abstractmethod
    def get_parameter_priors(self, args=None) -> Tuple[Tuple[pm.Distribution], pm.Distribution]:
        """ The parameter_priors """

    @ abstractmethod
    def get_mu_estimate(self, mu: Tuple[pm.Distribution]):
        """ The estimate of mu """

    @ abstractmethod
    def get_data_likelihood_type(self) -> Type[pm.Distribution]:
        """ The likelihood function type """

    # -- L --

    @ abstractmethod
    def load_data(self):
        """ Load the data """

    # -- S --

    def sample(self, args=None):
        model = self._create_model(args=args)
        traces = self._sample(model)
        return MCSamplerResult(traces, self)

    def _sample(self, model):
        vars_list = list(model.values_to_rvs.keys())[:-1]
        tune = draws = 2000
        # inference
        with model:
            traces = pm.sample(step=[pm.Slice(vars_list)], tune=tune, draws=draws)
        return traces

    def set_data(self, data):
        """ Set the data """
        self._cached_data = data
