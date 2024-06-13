# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pandas
from gws_core import (ConfigParams, FloatParam, InputSpec,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      task_decorator, OutputSpecs, InputSpecs, ListParam, BoolParam, BadRequestException)
from pandas import DataFrame

from ..table.ode_sim_result_table import ODESimResultTable, ODEStatus
from ..ode_system.ode_system import ODESystem
from ..helper.ode_system_helper import ODESystemHelper

@task_decorator("ODESimulator", human_name="ODE simulator",
                short_description="Dynamical simulation of systems of ordinary differential equations")
class ODESimulator(Task):
    """
    ODESimulator allows simulating dynamical systems given by ODE equations.
    """

    input_specs = InputSpecs({'system': InputSpec(ODESystem,
                                                  human_name="ODE system", short_description="The ODE system")})

    output_specs = OutputSpecs({'result': InputSpec(ODESimResultTable, human_name="ODE sim result table",
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
            default_value=100, human_name="Final time", short_description="The final simulation time"),
        'time_step':
        FloatParam(
            default_value=0.01, human_name="Time step", short_description="The simulation time step"),
        'initial_state':
        ListParam(
            default_value=None, optional=True, visibility=StrParam.PROTECTED_VISIBILITY,
            human_name="Initial state",
            short_description="The initial state vector (comma-separated floats)"),
        'method':
        StrParam(
            default_value="ODEINT",
            allowed_values=["ODEINT", "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
            human_name="Method or engine",
            short_description="Integration method. `ODEINT` use an integrator with a simpler interface based on lsoda from FORTRAN ODEPACK (it is faster)")}

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        ode_system = inputs["system"]

        t_start: float = params["initial_time"]
        t_end: float = params["final_time"]
        t_step: float = params["time_step"]
        method = params.get("method", "ODEINT_ENGINE")

        sim_system: ODESystem = ode_system.create_sim_system_helper()
        sol = sim_system.simulate(t_start, t_end, t_step=t_step, method=method)

        if method == "ODEINT_ENGINE":
            ode_status: ODEStatus = ODEStatus(success=True, message="")
        else:
            ode_status: ODEStatus = ODEStatus(success=sol.success, message=sol.message)

        t_df = DataFrame(data=sol.t, columns=["time"])
        y_df = DataFrame(data=sol.y, columns=sim_system.state_names())

        y = ODESimResultTable(data=pandas.concat([t_df, y_df], axis=1))
        y.set_ode_status(ode_status)

        return {"result": y}
