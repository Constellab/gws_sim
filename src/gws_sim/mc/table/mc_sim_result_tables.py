
# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import TypedDict

import pandas as pd
from gws_core import ResourceSet, TechnicalInfo, resource_decorator, Table
from ..helper.mc_system_helper import MCSystemHelperResult


@resource_decorator("MCSimResultTables")
class MCSimResultTables(ResourceSet):
    """ MCSimResultTables """

    PARAMETER_TRACE_TABLE_NAME = "Parameter traces"
    PREDICTION_MEANS_TABLE_NAME = "Prediction means"
    PREDICTION_STDS_TABLE_NAME = "Prediction stds"

    def __init__(self, result: MCSystemHelperResult):
        super().__init__()

        param_traces_table = Table(data=result.get_parameter_traces())
        param_traces_table.name = self.PARAMETER_TRACE_TABLE_NAME
        self.add_resource(param_traces_table)

        mean_df, std_df = result.get_state_prediction_stats()
        pred_means = Table(data=mean_df)
        pred_means.name = self.PREDICTION_MEANS_TABLE_NAME
        self.add_resource(pred_means)

        pred_stds = Table(data=std_df)
        pred_stds.name = self.PREDICTION_STDS_TABLE_NAME
        self.add_resource(pred_stds)

    def get_parameter_traces(self):
        return self.get_resource(self.PARAMETER_TRACE_TABLE_NAME)

    def get_prediction_means(self):
        return self.get_resource(self.PREDICTION_MEANS_TABLE_NAME)

    def get_prediction_stds(self):
        return self.get_resource(self.PREDICTION_STDS_TABLE_NAME)
