# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (Table, ConfigParams, FloatParam, InputSpec, IntParam,
                      Task, TaskInputs, TaskOutputs,
                      task_decorator, OutputSpecs, OutputSpec,InputSpecs, BoolParam)

from ..pinn_system.pinn_system import PINNSystem
from ..helper.pinn_system_helper import PINNSolution, PINNSystemHelper


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

    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="PINN sim result table",
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
            default_value=10, human_name="Final time", short_description="The final simulation time"),
        'number_hidden_layers':
        IntParam(
            default_value=40, human_name="Number of hidden layers", short_description=""),
        'width_hidden_layers':
        IntParam(
            default_value=5, human_name="Width of the hidden layers", short_description=""),
        'number_iterations':
        IntParam(
            default_value=2000, human_name="Interations", short_description=""),

    }

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        pinn_system: PINNSystem = inputs["system"]

        t_start: float = params["initial_time"]
        t_end: float = params["final_time"]
        number_hidden_layers: int = params["number_hidden_layers"]
        width_hidden_layers: int = params["width_hidden_layers"]
        number_iterations: int = params["number_iterations"]

        sim_system: PINNSystemHelper = pinn_system.create_sim_system_helper()
        sim_system.set_message_dispatcher(self.message_dispatcher)

        data_table: Table = inputs.get('data')

        sol: PINNSolution = sim_system.simulate(
            t_start, t_end, number_hidden_layers, width_hidden_layers, number_iterations, dataframe=data_table.get_data())

        if not sol.success:
            raise Exception(sol.message)

        res_table: Table = Table(sol.y)
        return {"result": res_table}
