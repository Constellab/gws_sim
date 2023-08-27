import json
import os

import numpy as np
from gws_core import BaseTestCase, File, Settings, Table, TaskRunner
from gws_sim import (ODESystem, ODESimResultTable, ODEStateTable)
from pandas import DataFrame

settings = Settings.get_instance()


class TestLoadsDumps(BaseTestCase):

    def test_ode_system_dump_load(self):
        pycode = """
from gws_sim import
class System():
    def initial_state(self, args=None):
        return [0, 1, 1.05]

    def parameters(self, t, args=None):
        return 10, 2.667, 28

    def derivative(self, t, x, args=None):
        u, v, w = x
        sigma, rho, beta = self.parameters(t, args)

        dudt = -sigma*(u - v)
        dvdt = rho*u - v - u*w
        dwdt = -beta*w + u*v
        return [dudt, dvdt, dwdt]
"""

        ode_sys = ODESystem(
            code=pycode
        )

        # dump as json
        code = ode_sys.dumps()
        self.assertTrue(isinstance(code, str))
        self.assertTrue("class System" in code)
        self.assertTrue("def parameters" in code)
        self.assertTrue("def initial_state" in code)

        sys2 = ode_sys.loads(code)
        self.assertTrue(isinstance(sys2, ODESystem))
