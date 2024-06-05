# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (Table, ConfigParams, OutputSpec, InputSpecs, InputSpec,
                      PythonCodeParam, Task, TaskInputs, TaskOutputs,
                      task_decorator, OutputSpecs)

from .pinn_system import PINNSystem

@task_decorator("PINNSystemBuilder", human_name="PINN system builder",
                short_description="The PINN system builder")
class PINNSystemBuilder(Task):

    output_specs = OutputSpecs({'system': OutputSpec(PINNSystem, human_name="PINN system",
                                                     short_description="The PINN system")})

    config_specs = {'code': PythonCodeParam(
        default_value="", human_name="PINN system code", short_description="Python code representing the PINN system")}

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        code = params.get("code")
        ode_sys = PINNSystem(code=code)
        return {'system': ode_sys}
