# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pandas as pd
from gws_core import (ConfigParams, InputSpec,
                      Task, TaskInputs, TaskOutputs,
                      task_decorator, OutputSpecs, InputSpecs, Table, IntParam)

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

    config_specs = {
        'draws':
        IntParam(
            default_value=2000, human_name="Draws per chain",
            short_description="The number of samples to draw. Defaults to 1000. The number of tuned samples are discarded by default"),
        'tune':
        IntParam(
            default_value=1000, visibility=IntParam.PROTECTED_VISIBILITY, human_name="Tune",
            short_description="The number of iterations to tune, defaults to 1000. Samplers adjust the step sizes, scalings or similar during tuning"),
        'chains':
        IntParam(
            default_value=2, visibility=IntParam.PROTECTED_VISIBILITY, human_name="Chains",
            short_description="The number of chains to sample")}

    _progress_tab = None

    # Define a callback function to get the progress value
    def progress_callback(self, trace, draw):
        current_chain = draw.chain
        self._progress_tab[current_chain] = 100 * trace.draw_idx/trace.draws
        average_value = sum(self._progress_tab)/len(self._progress_tab)
        self.update_progress_value(
            average_value, f"Chain {current_chain}: number of draws {trace.draw_idx} /{trace.draws}.")
        # print(f"{average_value}% ==> Chain {draw.chain}: number of draws {trace.draw_idx}/{trace.draws}")

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        data = inputs["data"]
        priors = inputs["priors"]
        mc_system = inputs["system"]

        # init helper
        sim_system_helper: MCSystemHelper = mc_system.create_sim_system_helper()
        sim_system_helper.set_data(data.get_data())
        sim_system_helper.set_priors(priors.get_data())

        draws = params["draws"]
        tune = params["tune"]
        chains = params["chains"]

        self._progress_tab = [0]*chains
        result = sim_system_helper.sample(tune=tune, draws=draws, chains=chains, callback=self.progress_callback)
        tables = MCSimResultTables(result=result)

        return {"result": tables}
