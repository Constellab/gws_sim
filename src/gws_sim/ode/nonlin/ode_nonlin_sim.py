# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import re

import numpy as np
import pandas
from gws_core import (BadRequestException, ConfigParams, FloatParam, InputSpec,
                      PythonCodeParam, StrParam, Task, TaskInputs, TaskOutputs,
                      task_decorator)
from pandas import DataFrame
from scipy.integrate import solve_ivp

from ..table.ode_sim_result_table import ODESimResultTable, ODEStatus
from .ode_nonlin_system import NonlinarODESystem


@task_decorator("NonlinearODESim", human_name="Nonlinear ODE simulator",
                short_description="Simulator of a system of monlinear ordinary differential equations")
class NonlinearODESim(Task):

    input_specs = {'system': InputSpec(NonlinarODESystem, is_optional=True, human_name="Linear ODE system",
                                       short_description="The Linear ODE system")}

    output_specs = {'result': InputSpec(ODESimResultTable, human_name="ODE sim result table",
                                        short_description="The table of simulation results")}

    config_specs = {
        'initial_time': FloatParam(default_value=0.0, human_name="Initial time", short_description="The initial simulation time"),
        'final_time': FloatParam(default_value=100, human_name="Final time", short_description="The final simulation time"),
        'time_step': FloatParam(default_value=0.01, human_name="Time step", short_description="The simulation time step"),
        'method': StrParam(default_value='RK45', allowed_values=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"], human_name="Method", short_description="Integration method"),
        'initial_state': PythonCodeParam(
            default_value="",
            optional=True, visibility=PythonCodeParam.PROTECTED_VISIBILITY,
            human_name="Intial state",
            short_description="The initial state (e.g. x = 10)"),
        'parameters':
        PythonCodeParam(
            default_value="",
            optional=True, visibility=PythonCodeParam.PROTECTED_VISIBILITY,
            human_name="Parameters",
            short_description="The parameters (e.g. alpha = -0.3)"),
        'equations':
        PythonCodeParam(
            default_value="",
            optional=True, visibility=PythonCodeParam.PROTECTED_VISIBILITY,
            human_name="Equation",
            short_description="The differential equations (e.g. dt/dx = alpha * x). It is not used if a nonlinear system is already given in the inputs."),
    }

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        if "system" in inputs:
            ode_system = inputs["system"]
        else:
            equations = params.get("equations")
            if equations:
                ode_system = NonlinarODESystem(equations=equations)
            else:
                raise BadRequestException(
                    "No equations defined. Please set equations in the inputs or in the parameters")

        initial_state = params.get("initial_state")
        if initial_state:
            ode_system.set_initial_state(initial_state)

        parameters = params.get("parameters")
        if parameters:
            ode_system.set_parameters(parameters)

        t_start: float = params["initial_time"]
        t_end: float = params["final_time"]
        if t_end <= t_start:
            raise BadRequestException("The final time must be greater than the initial time")

        t_step: float = params["time_step"]
        npoints = int((t_end-t_start) / t_step) + 1
        t_eval = np.linspace(t_start, t_end, num=npoints)

        u = []
        args = [u]
        method = params.get("method", "RK45")

        ode_system.initialize()
        sol = solve_ivp(
            fun=ode_system.derivative,
            t_span=[t_start, t_end],
            y0=ode_system.get_initial_state(as_float=True),
            method=method,
            t_eval=t_eval,
            args=args
        )

        ode_status: ODEStatus = ODEStatus(success=sol.success, message=sol.message)
        t_df = DataFrame(data=sol.t, columns=["time"])
        y_df = DataFrame(data=sol.y, index=ode_system.state_names).T
        y = ODESimResultTable(data=pandas.concat([t_df, y_df], axis=1))
        y.set_ode_status(ode_status)

        return {"result": y}
