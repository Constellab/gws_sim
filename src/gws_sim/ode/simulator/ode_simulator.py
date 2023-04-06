# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pandas
from gws_core import (BadRequestException, ConfigParams, FloatParam, InputSpec,
                      PythonCodeParam, StrParam, Task, TaskInputs, TaskOutputs,
                      task_decorator)
from pandas import DataFrame

from ..table.ode_sim_result_table import ODESimResultTable, ODEStatus
from ..ode_system.pycode_ode_system import PyCodeODESystem
from ..ode_system.simple_ode_system import SimpleODESystem
from ..sim_system.ode_sim_system import ODESimSystem


@task_decorator("ODESimulator", human_name="ODE simulator",
                short_description="Simulator of a system of ordinary differential equations")
class ODESimulator(Task):

    input_specs = {'system': InputSpec((PyCodeODESystem, SimpleODESystem),
                                       human_name="ODE system", short_description="The ODE system")}

    output_specs = {'result': InputSpec(ODESimResultTable, human_name="ODE sim result table",
                                        short_description="The table of simulation results")}

    config_specs = {
        'initial_time': FloatParam(default_value=0.0, human_name="Initial time", short_description="The initial simulation time"),
        'final_time': FloatParam(default_value=100, human_name="Final time", short_description="The final simulation time"),
        'time_step': FloatParam(default_value=0.01, human_name="Time step", short_description="The simulation time step"),
        'method': StrParam(default_value='RK45', allowed_values=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"], human_name="Method", short_description="Integration method")
    }

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        ode_system = inputs["system"]

        t_start: float = params["initial_time"]
        t_end: float = params["final_time"]
        t_step: float = params["time_step"]
        method = params.get("method", "RK45")

        sim_system: ODESimSystem = ode_system.create_sim_system()
        sol = sim_system.simulate(t_start, t_end, t_step=t_step, method=method)

        ode_status: ODEStatus = ODEStatus(success=sol.success, message=sol.message)
        t_df = DataFrame(data=sol.t, columns=["time"])
        y_df = DataFrame(data=sol.y, index=sim_system.state_names()).T
        y = ODESimResultTable(data=pandas.concat([t_df, y_df], axis=1))
        y.set_ode_status(ode_status)

        return {"result": y}
