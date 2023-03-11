# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import pandas
from gws_core import (BadRequestException, ConfigParams, FloatParam, InputSpec,
                      ListParam, StrParam, Task, TaskInputs, TaskOutputs,
                      task_decorator)
from pandas import DataFrame
from scipy.integrate import solve_ivp

from ..lin.ode_lin_system import LinarODESystem
from ..table.ode_sim_result_table import ODESimResultTable, ODEStatus


@task_decorator("LinearODESim", human_name="Linear ODE simulator",
                short_description="Simulator of a system of linear ordinary differential equations")
class LinearODESim(Task):

    input_specs = {'system': InputSpec(LinarODESystem, human_name="Linear ODE system",
                                       short_description="The Linear ODE system")}
    output_specs = {'result': InputSpec(ODESimResultTable, human_name="ODE sim result table",
                                        short_description="The table of simulation results")}
    config_specs = {
        'initial_state': ListParam(default_value=[], optional=True, human_name="Initial state", short_description="The initial state"),
        'initial_time': FloatParam(default_value=0.0, human_name="Initial time", short_description="The initial simulation time"),
        'final_time': FloatParam(default_value=100, human_name="Final time", short_description="The final simulation time"),
        'time_step': FloatParam(default_value=0.01, human_name="Time step", short_description="The simulation time step"),
        'method': StrParam(default_value='RK45', allowed_values=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"], human_name="Method", short_description="Integration method"),
    }

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        ode_system = inputs["system"]
        init_state = [float(x) for x in params["initial_state"]]
        if init_state:
            ode_system.set_initial_state(init_state)

        t_start: float = params["initial_time"]
        t_end: float = params["final_time"]
        if t_end <= t_start:
            raise BadRequestException("The final time must be greater than the initial time")

        t_step: float = params["time_step"]
        npoints = int((t_end-t_start) / t_step) + 1
        t_eval = np.linspace(t_start, t_end, num=npoints)

        method = params.get("method", "RK45")

        u = []
        args = [u]

        ode_system.initialize()
        sol = solve_ivp(
            fun=ode_system.derivative,
            t_span=[t_start, t_end],
            y0=ode_system.get_initial_state(),
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
