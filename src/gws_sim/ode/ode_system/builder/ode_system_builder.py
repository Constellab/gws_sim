# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, OutputSpec,
                      PythonCodeParam, StrParam, Task, TaskInputs, TaskOutputs,
                      task_decorator)

from ..simple_ode_system import SimpleODESystem
from ..pycode_ode_system import PyCodeODESystem


@task_decorator("ODESystemBuilder", human_name="ODE system builder",
                short_description="The ODE system builder")
class ODESystemBuilder(Task):

    input_specs = {}
    output_specs = {'system': OutputSpec((SimpleODESystem, PyCodeODESystem), human_name="ODE system",
                                         short_description="The ODE system")}

    config_specs = {
        'content_type':
        StrParam(
            default_value="plain", allowed_values=["plain", "pycode"],
            human_name="Content type", short_description="The content type"),
        'content_value': PythonCodeParam(default_value="", human_name="Content text or code ", short_description="Content text or code")}

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        content_type = params.get("content_type")
        content_value = params.get("content_value")
        if content_type == "plain":
            ode_sys = SimpleODESystem.from_text(content_value)
        else:
            ode_sys = PyCodeODESystem(code=content_value)
        return {'system': ode_sys}
