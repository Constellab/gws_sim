# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Tuple

import pymc as pm
from pandas import DataFrame

from .mc_sampler import MCSampler


class MCODEParamSampler(MCSampler):

    _cached_time_data = None
    _cached_nb_state = None

    def get_mu_estimate(self, mu: Tuple[pm.Distribution]):
        # extract time data
        if self._cached_time_data is None:
            self._cached_time_data = self.get_time_data().values.tolist()

        t_eval = self._cached_time_data
        t_start = t_eval[0]
        t_end = t_eval[-1]

        # extract the parameters and states
        nb_par = len(mu) - self.get_nb_states()

        parameters = mu[0: (nb_par+1)]
        initial_state = mu[nb_par:]

        sys = self.get_sim_system()
        sol = sys.simulate(
            t_start,
            t_end,
            initial_state=initial_state,
            parameters=parameters,
            t_step=None,
            t_eval=t_eval,
            args=None
        )
        return sol.y

    def get_observed_data(self):
        """ Returns the observed data """
        return self.get_data().iloc[:, 1:]

    def get_time_data(self) -> DataFrame:
        """ Returns time data """
        return self.get_data().iloc[:, 0]

    def get_state_data(self) -> DataFrame:
        """ Returns state data """
        return self.get_observed_data()

    def get_nb_states(self) -> int:
        """ Returns the number of states """
        if self._cached_nb_state is None:
            self._cached_nb_state = self.get_state_data().shape[1]
        return self._cached_nb_state
