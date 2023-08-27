# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod
from typing import List, Tuple, Type, Dict

from pandas import DataFrame
import numpy as np
import pandas as pd

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

from gws_core import BadRequestException
# from ...helper.base_sim_system_helper import BaseSimSystemHelper


class MCSystemHelperResult:
    """ Linear regression data """
    _traces = None
    _mc_system_helper: 'MCSystemHelper' = None
    _cached_predictions = None

    def __init__(self, traces, sampler):
        self._traces = traces
        self._mc_system_helper = sampler

    def get_traces(self):
        return self._traces

    def get_parameter_traces(self, num_samples=None):
        dfs = []
        for colname in self._mc_system_helper.get_param_names():
            data = az.extract(self._traces, num_samples=num_samples, group='posterior',
                              combined=True, var_names=colname)
            dfs.append(DataFrame(data.T, columns=[colname]))
        return pd.concat(dfs, axis=1)

    def get_time_data(self):
        return self._mc_system_helper.get_time_data()

    def get_state_predictions(self, num_samples=None) -> Dict[str, DataFrame]:
        if self._cached_predictions is not None:
            return self._cached_predictions

        param_traces: DataFrame = self.get_parameter_traces(num_samples=num_samples)
        num_samples = param_traces.shape[0]
        preds = None

        state_names = self._mc_system_helper.get_state_data().columns

        for i in range(0, num_samples):
            mu = param_traces.iloc[i, :].values.tolist()
            params = mu[0: -1]                              # do not keep `sigma`
            sol = self._mc_system_helper.get_mu_estimate(params)     # solve ode with the parameters
            m, n = sol.shape

            if preds is None:
                # allocate mem
                preds = {}
                for k in range(n):
                    name = state_names[k]
                    preds[name] = np.zeros(shape=(m, num_samples))

            for k in range(n):
                name = state_names[k]
                preds[name][:, i] = sol[:, k]

        self._cached_predictions = preds

        return preds

    def get_state_prediction_stats(self, num_samples=None):
        preds: Dict[str, DataFrame] = self.get_state_predictions(num_samples=num_samples)

        means = []
        stds = []
        for name, data in preds.items():
            stds.append(DataFrame(data=data.std(axis=1), columns=[name]))
            means.append(DataFrame(data=data.mean(axis=1),  columns=[name]))

        means = pd.concat(means, axis=1)
        stds = pd.concat(stds, axis=1)

        return means, stds


class MCSystemHelper:
    _cached_data: DataFrame = None
    _cached_prior_data: Dict[str, Dict] = None
    _cached_prior_dists: List = None
    _cached_param_names: List[str] = None

    # -- C --

    def _create_model(self, args=None):

        @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
        def pytensor_forward_model_matrix(mu):
            return self.get_mu_estimate(mu)

        with pm.Model() as model:
            mu, likelihood = self.get_priors_as_dists(args=args)
            mu_hat = pytensor_forward_model_matrix(pm.math.stack(mu))
            likelihood_type = likelihood["type"]
            likelihood_type(
                "y_obs",
                mu=mu_hat,
                sigma=likelihood["sigma"],
                observed=self.get_state_data()
            )
            return model

    # -- G --

    def get_data(self):
        """ Returns the data """
        if self._cached_data is None:
            data = self.load_data()
            if data is None:
                raise BadRequestException("No data defined. Please set data or define the `load_data()` method")

            self._cached_data = data
        return self._cached_data

    def get_time_data(self):
        """ Returns the time """
        return self.get_data().iloc[:, 0]

    def get_state_data(self):
        """ Returns the observed state data """
        return self.get_data().iloc[:, 1:]

    def get_param_names(self) -> List[str]:
        if self._cached_param_names is None:
            prior_data: Dict = self.get_parameter_priors()
            mu_data = prior_data["mu"]
            likelihood_data = prior_data["likelihood"]
            self._cached_param_names = [
                *[data["name"] for data in mu_data],
                *[data["name"] for data in likelihood_data["params"]],
            ]
        return self._cached_param_names

    def get_priors_as_dists(self, args=None) -> Tuple[Tuple[pm.Distribution], pm.Distribution]:
        if self._cached_prior_dists is None:
            prior_data: Dict = self.get_parameter_priors()

            # mu
            mu_data = prior_data["mu"]
            mu = []
            for data in mu_data:
                name = data["name"]
                mu.append(pm.TruncatedNormal(
                    name,
                    mu=data["mu"],
                    sigma=data["sigma"],
                    lower=data["lower"],
                    initval=data.get("initval", data["mu"])
                ))

            # likelihood
            likelihood_data = prior_data["likelihood"]
            likelihood_type = likelihood_data["type"]
            if likelihood_type == "Normal":
                likelihood_type = pm.Normal
            likelihood_params = likelihood_data["params"][0]
            sigma = pm.HalfNormal("sigma", sigma=likelihood_params["sigma"])

            likelihood = {"type": likelihood_type, "sigma": sigma}
            self._cached_prior_dists = (mu, likelihood)

        return self._cached_prior_dists

    def get_parameter_priors(self, args=None) -> Tuple[Tuple[pm.Distribution], pm.Distribution]:
        """ The parameter priors """

        if self._cached_prior_data is None:
            self._cached_prior_data = self.load_priors()

        return self._cached_prior_data

    @ abstractmethod
    def get_mu_estimate(self, mu: Tuple[pm.Distribution]):
        """ The estimate of mu """

    # -- L --

    @ abstractmethod
    def load_data(self):
        """ Load the data """

    @ abstractmethod
    def load_priors(self) -> List[Dict]:
        """ Load priors """

    # -- S --

    def sample(self, args=None):
        model = self._create_model(args=args)
        traces = self._sample(model)
        return MCSystemHelperResult(traces, self)

    def _sample(self, model):
        vars_list = list(model.values_to_rvs.keys())[:-1]
        tune = draws = 2000
        # inference
        with model:
            traces = pm.sample(step=[pm.Slice(vars_list)], tune=tune, draws=draws)
        return traces

    def set_data(self, data: DataFrame):
        """ Set the data """
        self._cached_data = data

    def set_priors(self, priors: Dict[str, Dict]):
        """ Set the priors """
        self._cached_prior_data = priors
