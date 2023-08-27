
import os
from gws_core import BaseTestCase, TaskRunner, File, Settings
from gws_sim import ODESimulator, ODESystemImporter


class TestODESimualtor(BaseTestCase):

    def test_ode_lorentz(self):
        settings = Settings.get_instance()
        data_dir = settings.get_variable("gws_sim:testdata_dir")

        sys = ODESystemImporter.call(
            File(path=os.path.join(data_dir, "ode_lorentz", "_ode_lorentz_system.py")),
            {}
        )

        tester = TaskRunner(
            params={
                "initial_time": 0.0,
                "initial_state": [0, 1, "1.05"],
                "time_step": 1.0,
                "final_time": 500,
                "method": "RK45"},
            inputs={"system": sys},
            task_type=ODESimulator
        )
        outputs = tester.run()
        table = outputs["result"]

        df = table.get_data()
        print(df)
        self.assertEqual(df.shape, (501, 4))
        self.assertEqual(df.iat[0, 0], 0.0)
        self.assertEqual(df.iat[1, 0], 1.0)
        self.assertEqual(df.iat[0, 1], 0.0)
        self.assertEqual(df.iat[500, 0], 500)
        print(table)
