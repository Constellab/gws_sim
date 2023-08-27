
import os
import numpy as np
import pandas as pd
import json
from gws_core import BaseTestCase, TaskRunner, File, Settings, TableImporter
from gws_sim import MCSimulator, MCSystemImporter, MCPriorDictImporter


class TestODESimualtor(BaseTestCase):

    def test_mc_lv(self):
        settings = Settings.get_instance()
        data_dir = settings.get_variable("gws_sim:testdata_dir")

        sys = MCSystemImporter.call(File(path=os.path.join(data_dir, "mc_lv", "_mc_lv_system_lite.py")), {})
        priors = MCPriorDictImporter.call(File(path=os.path.join(data_dir, "mc_lv", "lv_priors.json")), {})
        data = TableImporter.call(File(path=os.path.join(data_dir, "mc_lv", "lv_data.csv")), {})

        # =================== PRIORS ===================

        tester = TaskRunner(
            params={},
            inputs={
                "data": data,
                "priors": priors,
                "system": sys
            },
            task_type=MCSimulator
        )
        outputs = tester.run()
        result = outputs["result"]

        traces = result.get_parameter_traces()
        pred_means = result.get_prediction_means()
        pred_stds = result.get_prediction_stds()

        print(traces)
        print(pred_means)
        print(pred_stds)

        self.assertEqual(pred_means.shape, (21, 2))
        self.assertEqual(pred_stds.shape, (21, 2))
