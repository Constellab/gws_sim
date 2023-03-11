import json
import os

import numpy as np
from gws_core import BaseTestCase, File, Settings, Table, TaskRunner
from gws_sim import (NonlinarODESystem, NonlinearODESim, ODESimResultTable,
                     ODEStateTable)
from pandas import DataFrame

settings = Settings.get_instance()


class TestLoadsDumps(BaseTestCase):

    def test_nonlin_exporter(self):
        ode_sys = NonlinarODESystem(
            equations=DataFrame(
                data=["dv/dt = -0.3 * v", "dp/dt = -0.3 * p"],
                columns=["equation"],
                index=["velocity", "position"])
        )

        data1 = ode_sys.dumps()
        self.assertEqual(
            json.dumps(data1),
            '{"equations": {"velocity": "dv/dt = -0.3 * v", "position": "dp/dt = -0.3 * p"}, "default_initial_state": [], "default_parameters": []}')

        print(json.dumps(data1))

        new_ode_sys = ode_sys.loads(data1)
        data2 = new_ode_sys.dumps()
        self.assertEqual(data1, data2)

    def test_nonlin2_exporter(self):
        data_dir = settings.get_variable("gws_ode:testdata_dir")

        ode_sys = NonlinarODESystem(
            equations=DataFrame(
                data=["dv/dt = law.hill(vmax,KH,n,v)", "dp/dt = -0.3 * p"],
                columns=["equation"],
                index=["glucose", "fructose"])
        )
        ode_sys.set_default_parameters("vmax,KH,n = (2,0.5,3)")
        ode_sys.set_default_initial_state("v,p=(5,6)")

        data1 = ode_sys.dumps()
        with open(os.path.join(data_dir, "./ode/toy_ode/toy_ode.json"), "w", encoding="utf-8") as fp:
            json.dump(data1, fp, indent=4)

        self.assertEqual(
            json.dumps(data1),
            '{"equations": {"glucose": "dv/dt = law.hill(vmax,KH,n,v)", "fructose": "dp/dt = -0.3 * p"}, "default_initial_state": ["v,p=(5,6)"], "default_parameters": ["vmax,KH,n = (2,0.5,3)"]}')

        print(json.dumps(data1))

        new_ode_sys = ode_sys.loads(data1)
        data2 = new_ode_sys.dumps()
        self.assertEqual(data1, data2)

    def test_lorentz_exporter(self):
        data_dir = settings.get_variable("gws_ode:testdata_dir")

        ode_sys = NonlinarODESystem(
            equations=DataFrame(
                data=["du/dt = -sigma*(u - v)", "dv/dt = rho*u - v - u*w", "dw/dt = -beta*w + u*v"],
                columns=["equation"],
                index=["u", "v", "w"])
        )
        ode_sys.set_default_parameters("sigma, beta, rho = 10, 2.667, 28")
        ode_sys.set_default_initial_state("u, v, w = 0, 1, 1.05")

        data1 = ode_sys.dumps()
        print(os.path.join(data_dir, "./ode/lorentz/lorentz.json"))
        with open(os.path.join(data_dir, "./ode/lorentz/lorentz.json"), "w", encoding="utf-8") as fp:
            json.dump(data1, fp, indent=4)
