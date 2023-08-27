import os

import numpy as np
from gws_core import BaseTestCase, File, Settings, Table, TaskRunner
from gws_sim import MCSystem


class TestLoadsDumps(BaseTestCase):

    def test_ode_system_dump_load(self):
        settings = Settings.get_instance()
        data_dir = settings.get_variable("gws_sim:testdata_dir")

        path = os.path.join(data_dir, "mc_lv", "_mc_lv_system_full.py")
        with open(path, mode="r", encoding="utf-8") as fp:
            pycode = fp.read()

        mc_sys = MCSystem(
            code=pycode
        )

        # dump as json
        code = mc_sys.dumps()
        self.assertTrue(isinstance(code, str))
        sys2 = mc_sys.loads(code)
        self.assertTrue(isinstance(sys2, MCSystem))
