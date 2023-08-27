import os
from gws_core import BaseTestCase, File, Settings, Table, TaskRunner

from gws_core import BaseTestCase
from gws_sim import MCSystem, MCSystemHelper


class TestMCSystem(BaseTestCase):

    def test_ode_lorentz(self):

        settings = Settings.get_instance()
        data_dir = settings.get_variable("gws_sim:testdata_dir")

        path = os.path.join(data_dir, "mc_lv", "_mc_lv_system_full.py")
        with open(path, mode="r", encoding="utf-8") as fp:
            pycode = fp.read()

        mc_sys = MCSystem(
            code=pycode
        )

        sim_sys = mc_sys.create_sim_system_helper()
        self.assertTrue(isinstance(sim_sys, MCSystemHelper))

        result = sim_sys.sample()

        print(result.get_parameter_traces())
        print(result.get_state_predictions())
        print(result.get_state_prediction_stats())
