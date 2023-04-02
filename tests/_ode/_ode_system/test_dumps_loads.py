import json
import os

import numpy as np
from gws_core import BaseTestCase, File, Settings, Table, TaskRunner
from gws_sim import (PyCodeODESystem, SimpleODESystem, ODESimResultTable, ODEStateTable)
from pandas import DataFrame

settings = Settings.get_instance()


class TestLoadsDumps(BaseTestCase):

    def test_simple_ode_system_dump_load(self):
        ode_sys = SimpleODESystem(
            equations=["du/dt = -sigma*(u - v)", "dv/dt = rho*u - v - u*w", "dw/dt = -beta*w + u*v"],
            default_parameters=["sigma, beta, rho = 10, 2.667, 28"],
            default_initial_state=["u, v, w = 0, 1, 1.05"]
        )

        # dump as json
        data = ode_sys.dumps()
        self.assertTrue(isinstance(data, dict))
        self.assertTrue("default_parameters" in data)
        self.assertTrue("default_initial_state" in data)
        self.assertEqual(data["default_parameters"], ['sigma, beta, rho = 10, 2.667, 28'])

        sys2 = ode_sys.loads(data)
        self.assertTrue(isinstance(sys2, SimpleODESystem))

        # dump as test
        text_data = ode_sys.dumps(text=True)
        self.assertTrue(isinstance(text_data, str))
        self.assertTrue("#default_parameters" in text_data)

        sys3 = ode_sys.loads(text_data)
        self.assertTrue(isinstance(sys3, SimpleODESystem))

    def test_pycode_ode_system_dump_load(self):
        pycode = """
from gws_sim import SimSystem
class Model(SimSystem):
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

        ode_sys = PyCodeODESystem(
            code=pycode
        )

        # dump as json
        code = ode_sys.dumps()
        self.assertTrue(isinstance(code, str))
        self.assertTrue("class Model" in code)
        self.assertTrue("def parameters" in code)
        self.assertTrue("def initial_state" in code)

        sys2 = ode_sys.loads(code)
        self.assertTrue(isinstance(sys2, PyCodeODESystem))
