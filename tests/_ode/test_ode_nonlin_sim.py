import json
import os

import numpy as np
from gws_core import BaseTestCase, File, Settings, Table, TaskRunner
from gws_sim import (NonlinarODESystem, NonlinearODESim, ODESimResultTable,
                     ODEStateTable)
from pandas import DataFrame

settings = Settings.get_instance()


class TestNonlinearODESim(BaseTestCase):

    async def test_ode_sim_1d_nonlin(self):
        ode_sys = NonlinarODESystem(
            equations=DataFrame(data=["dv/dt = -0.3 * v"], columns=["equation"], index=["velocity"])
        )
        # init_state = ODEInitialStateTable(data=[5], row_names=["foo"])

        tester = TaskRunner(
            params={"initial_state": "v = 5", "final_time": 20},
            inputs={"system": ode_sys},
            task_type=NonlinearODESim
        )
        outputs = await tester.run()

        # test results
        result = outputs["result"]

        df = result.get_data()
        print(df)

        self.assertEqual(df.iloc[0, 0], 0.00)
        self.assertEqual(df.iloc[0, 1], 5.00)

        self.assertEqual(df.iloc[1, 0], 0.01)
        self.assertTrue(np.all(np.isclose(df.iloc[1, 1], 4.985022, atol=0.001)))

        self.assertEqual(df.iloc[4, 0], 0.04)
        self.assertTrue(np.all(np.isclose(df.iloc[4, 1], 4.940359, atol=0.001)))
        print(result)

    async def test_ode_sim_2d_nonlin(self):
        ode_sys = NonlinarODESystem(
            equations=DataFrame(
                data=["dv/dt = -0.3 * v", "dp/dt = -0.3 * p"],
                columns=["equation"],
                index=["velocity", "position"])
        )

        tester = TaskRunner(
            params={"initial_state": "v,p = (5,5)", "final_time": 20},
            inputs={"system": ode_sys},
            task_type=NonlinearODESim
        )
        outputs = await tester.run()

        # test results
        result = outputs["result"]

        df = result.get_data()
        self.assertEqual(df.shape, (2001, 3))

        self.assertEqual(df.iloc[0, 0], 0.00)
        self.assertEqual(df.iloc[0, 1], 5.00)

        self.assertEqual(df.iloc[1, 0], 0.01)
        self.assertTrue(np.all(np.isclose(df.iloc[1, 1], 4.985022, atol=0.001)))

        self.assertEqual(df.iloc[4, 0], 0.04)
        self.assertTrue(np.all(np.isclose(df.iloc[4, 1], 4.940359, atol=0.001)))

        self.assertTrue(np.all(np.isclose(df.iloc[:, 1].values, df.iloc[:, 2].values, atol=0.001)))
        print(result)

    async def test_ode_sim_1d_hill(self):
        ode_sys = NonlinarODESystem(
            equations=DataFrame(
                data=["dv/dt = law.hill(2,0.5,3,v)"],
                columns=["equation"],
                index=["glucose"])
        )

        tester = TaskRunner(
            params={"initial_state": "v = 5", "final_time": 20},
            inputs={"system": ode_sys},
            task_type=NonlinearODESim
        )
        outputs = await tester.run()

        # test results
        result = outputs["result"]
        df = result.get_data()
        self.assertEqual(df.iloc[4, 0], 0.04)
        self.assertTrue(np.all(np.isclose(df.iloc[4, 1], 5.079922, atol=0.001)))
        print(df)

    async def test_ode_sim_1d_hill_with_param(self):
        ode_sys = NonlinarODESystem(
            equations=DataFrame(
                data=["dv/dt = law.hill(vmax,KH,n,v)"],
                columns=["equation"],
                index=["glucose"])
        )

        tester = TaskRunner(
            params={
                "initial_state": "v=5",
                "final_time": 20,
                "parameters": "vmax,KH,n = (2,0.5,3)"  # "vmax=2 \nKH=0.5 \nn=3"
            },
            inputs={"system": ode_sys},
            task_type=NonlinearODESim
        )
        outputs = await tester.run()

        # test results
        result = outputs["result"]
        df = result.get_data()
        self.assertEqual(df.iloc[4, 0], 0.04)
        self.assertTrue(np.all(np.isclose(df.iloc[4, 1], 5.079922, atol=0.001)))
        print(df)


    async def test_ode_lorentz(self):
        ode_sys = NonlinarODESystem(
            equations=DataFrame(
                data=["du/dt = -sigma*(u - v)", "dv/dt = rho*u - v - u*w", "dw/dt = -beta*w + u*v"],
                columns=["equation"],
                index=["u", "v", "w"])
        )

        tester = TaskRunner(
            params={
                "initial_state": "u, v, w = 0, 1, 1.05",
                "final_time": 100,
                "parameters": "sigma, beta, rho = 10, 2.667, 28"
            },
            inputs={"system": ode_sys},
            task_type=NonlinearODESim
        )
        outputs = await tester.run()

        # test results
        result = outputs["result"]
        df = result.get_data()
        print(df)
