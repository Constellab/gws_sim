# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pandas
from gws_core import (Table, ConfigParams, FloatParam, InputSpec,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      task_decorator, OutputSpecs, InputSpecs, ListParam, BoolParam, BadRequestException)
from pandas import DataFrame

from ..pinn_system.pinn_system import PINNSystem
from ..helper.pinn_system_helper import PINNSolution


@task_decorator("PINNSimulator", human_name="PINN simulator",
                short_description="Dynamical simulation of systems of ordinary differential equations")
class PINNSimulator(Task):
    """
    ODESimulator allows simulating dynamical systems given by ODE equations.
    """
    
    input_specs = InputSpecs({
        'system': InputSpec(PINNSystem, human_name="ODE system", short_description="The ODE system"),
        'data': InputSpec(Table, human_name='Data', short_description='Dataset')
    })

    output_specs = OutputSpecs({'result': InputSpec(Table, human_name="PINN sim result table",
                                                    short_description="The table of simulation results")})

    config_specs = {
        'predictive_controller':
        BoolParam(
            default_value=False),
        'initial_time':
        FloatParam(
            default_value=0.0, human_name="Initial time", short_description="The initial simulation time"),
        'final_time':
        FloatParam(
            default_value=100, human_name="Final time", short_description="The final simulation time")
    }

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        pinn_system = inputs["system"]

        t_start: float = params["initial_time"]
        t_end: float = params["final_time"]

        sim_system: PINNSystem = pinn_system.create_sim_system_helper()

        data_table: Table = inputs.get('data')

        sol: PINNSolution = sim_system.simulate(
            t_start, t_end, dataframe=data_table.get_data())

        if not sol.success:
            raise Exception(sol.message)

        res_table: Table = Table(sol.y)
        return {"result": res_table}
