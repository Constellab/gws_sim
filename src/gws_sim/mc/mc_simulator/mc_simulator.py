# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pandas as pd
from gws_core import (ConfigParams, InputSpec,
                      Task, TaskInputs, TaskOutputs,
                      task_decorator, OutputSpecs, InputSpecs, Table)

from ..table.mc_sim_result_tables import MCSimResultTables
from ..mc_system.mc_system import MCSystem
from ..helper.mc_system_helper import MCSystemHelper
from ..mc_prior_dict.mc_prior_dict import MCPriorDict


@task_decorator("MCSimulator", human_name="MC simulator",
                short_description="Monte-Carlo fitting and simulation of ODE models using bayesian inference")
class MCSimulator(Task):
    """
    MCSimulator allows fitting and simulating ODE models using Monte-Carlo simulations.

    Bayesian inference is used to fit ODE models to experimental data.
    The inference is performed by sampling models' parameters using the Monte-Carlo libray PyMC (https://www.pymc.io/)
    """

    input_specs = InputSpecs({
        'data': InputSpec(Table, human_name="Data table", short_description="The data table"),
        'priors': InputSpec(MCPriorDict, human_name="Parameter priors", short_description="The parameter priors"),
        'system': InputSpec(MCSystem, human_name="MC system", short_description="The MC system")
    })

    output_specs = OutputSpecs({'result': InputSpec(MCSimResultTables, human_name="MC sim result tables",
                                                    short_description="The table of simulation results")})

    config_specs = {}

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        data = inputs["data"]
        priors = inputs["priors"]
        mc_system = inputs["system"]

        # init helper
        sim_system_helper: MCSystemHelper = mc_system.create_sim_system_helper()
        sim_system_helper.set_data(data.get_data())
        sim_system_helper.set_priors(priors.get_data())

        result = sim_system_helper.sample()
        tables = MCSimResultTables(result=result)

        return {"result": tables}
