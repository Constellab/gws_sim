import json
import os

import numpy as np
from gws_core import BaseTestCase, File, Settings, Table, TaskRunner
from gws_sim import (SimSystem, ODEStateTable)
from pandas import DataFrame

settings = Settings.get_instance()


class Lorentz(SimSystem):
    def initial_state(self, args=None):
        """ initial_state """
        return [0, 1, 1.05]

    def parameters(self, t, args=None):
        """ parameters """
        return 10, 2.667, 28

    def derivative(self, t, x, args=None):
        """ derivative """
        u, v, w = x
        sigma, rho, beta = self.parameters(t, args)

        dudt = -sigma*(u - v)
        dvdt = rho*u - v - u*w
        dwdt = -beta*w + u*v
        return [dudt, dvdt, dwdt]


class TestLinearODESim(BaseTestCase):

    def test_nonlin(self):
        sys = Lorentz()
        sol = sys.simulate(t_start=0, t_end=100, t_step=0.05)

        self.assertEqual(sol.success, True)
        self.assertEqual(sol.status, 0)

        print(sol)
