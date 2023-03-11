import json
import os

import numpy as np
from gws_core import BaseTestCase, File, Settings, Table, TaskRunner
from gws_sim import (LinarODESystem, LinearODESim, ODESimResultTable,
                     ODEStateTable)
from pandas import DataFrame

settings = Settings.get_instance()


class TestLinearODESim(BaseTestCase):

    async def test_ode_sim_1d_lin(self):
        ode_sys = LinarODESystem(
            A=DataFrame(data=[-0.3], columns=["velocity"], index=["velocity"]),
            B=None
        )
        # init_state = ODEInitialStateTable(data=[5], row_names=["foo"])

        tester = TaskRunner(
            params={"initial_state": [5], "final_time": 20},
            inputs={"system": ode_sys},
            task_type=LinearODESim
        )
        outputs = await tester.run()

        # test results
        result = outputs["result"]

        df = result.get_data()

        self.assertEqual(df.iloc[0, 0], 0.00)
        self.assertEqual(df.iloc[0, 1], 5.00)

        self.assertEqual(df.iloc[1, 0], 0.01)
        self.assertTrue(np.all(np.isclose(df.iloc[1, 1], 4.985022, atol=0.001)))

        self.assertEqual(df.iloc[4, 0], 0.04)
        self.assertTrue(np.all(np.isclose(df.iloc[4, 1], 4.940359, atol=0.001)))
        print(result)

    async def test_ode_sim_2d_lin(self):
        ode_sys = LinarODESystem(
            A=DataFrame(
                data=[[-0.3, 0,],
                      [0, -0.3]],
                columns=["velocity", "position"],
                index=["velocity", "position"]),
            B=None)
        # init_state = ODEStateTable(data=[5, 5])

        tester = TaskRunner(
            params={"initial_state": [5, 5], "final_time": 20},
            inputs={"system": ode_sys},
            task_type=LinearODESim
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
