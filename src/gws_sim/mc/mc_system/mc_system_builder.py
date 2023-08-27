# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, OutputSpec,
                      PythonCodeParam, Task, TaskInputs, TaskOutputs,
                      task_decorator, OutputSpecs)

from .mc_system import MCSystem


@task_decorator("MCSystemBuilder", human_name="MC system builder",
                short_description="The MC system builder")
class MCSystemBuilder(Task):
    """
    Creates a MCSystem using a raw Python code.

    The output MCSystem can then be used for dynamical simulations.

    A typical MC code is built using the MCSystemHelper class provided in Constellab. It looks like

        ```
        from gws_sim import MCSystemHelper
        class Model(MCSystemHelper):
            def initial_state(self, args=None):
                return [0, 1, 1.05]

            def parameters(self, t, args=None):
                return 10, 2.667, 28

            def derivative(self, t, x, args=None):
                u, v, w = x
                sigma, rho, beta = self.parameters(t, args)

                dudt = -sigma*(u - v)
                dvdt = rho*u - v - u*w
                dwdt = -beta*w + u*v
                return [dudt, dvdt, dwdt]
        ```
    """
    output_specs = OutputSpecs({'system': OutputSpec(MCSystem, human_name="MC system",
                                                     short_description="The MC system")})

    config_specs = {'code': PythonCodeParam(
        default_value="", human_name="ODE system code", short_description="Python code representing the MC system")}

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        code = params.get("code")
        ode_sys = MCSystem(code=code)
        return {'system': ode_sys}
