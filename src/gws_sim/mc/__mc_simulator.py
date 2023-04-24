# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pandas
from gws_core import (ConfigParams, FloatParam, InputSpec,IntParam,
                      PythonCodeParam, StrParam, Task, TaskInputs, TaskOutputs,
                      task_decorator, Table)
from pandas import DataFrame
from scipy.optimize import least_squares
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import tensorflow as tf
from pytensor.compile.ops import as_op
from scipy.integrate import odeint
from numba import njit

from ..table.ode_sim_result_table import ODESimResultTable, ODEStatus
from ..ode_system.pycode_ode_system import PyCodeODESystem
from ..ode_system.simple_ode_system import SimpleODESystem


from ..sim_system.mcmc_system import MCMCSystem


@task_decorator("MCMCSimulator", human_name="MCMC simulator",
                short_description="Simulator of a system of ordinary differential equations using MCMC")
class MCMCSimulator(Task):

    input_specs = {'system': InputSpec((PyCodeODESystem, SimpleODESystem),
                                       human_name="ODE system", short_description="The ODE system"),
                    'data': InputSpec(Table, human_name="Data Table", short_description="The table of the experimental data")}

    output_specs = {'result': InputSpec(ODESimResultTable, human_name="ODE sim result table",
                                        short_description="The table of simulation results")}

    config_specs = {
        'initial_time': FloatParam(default_value=0.0, human_name="Initial time", short_description="The initial simulation time"),
        'final_time': FloatParam(default_value=20, human_name="Final time", short_description="The final simulation time"),
        'time_step': FloatParam(default_value=1, human_name="Time step", short_description="The simulation time step"),
        'method': StrParam(default_value='RK45', allowed_values=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"], human_name="Method", short_description="Integration method"),
        'tune': FloatParam(default_value=1000, human_name="Number of iterations", short_description="The number of iterations to tune"),
        'draws': FloatParam(default_value=1000, human_name="Number of samples", short_description="The number of samples to draw"),
        'cores': IntParam(default_value=4, human_name="Core number", short_description="The number of core for estimation"),
        'prior_type': StrParam(human_name="priorType", short_description="Type of prior", default_value='normal', allowed_values=["normal", "uniform", "halfnormal", "halfcauchy", "truncatednormal"],)

    }

    # va falloir l'adapter  data-solution
    def ode_model_residuals(self,theta, data, solution):
        names = data.column_names
        #TODO : pourquoi on inverse les colonnes de hare et lynx ?
        data_column = np.array([data.get_column_data(names[2]), data.get_column_data(names[1])])
        solution_column = np.array([solution.get_column_data("x"),solution.get_column_data("y")])
        soustraction = data_column - solution_column
        soustraction_flat = soustraction.flatten(order="F")
        return (soustraction_flat)

    # #TODO : changer la fonction odeint
    # # decorator with input and output types a Pytensor double float tensors
    # @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    # def pytensor_forward_model_matrix(self):
    #     #sim_system = self.sim_system
    #     theta = self.theta
    #     data = self.data
    #     names = data.column_names
    #     #result_py = odeint(func=sim_system.derivative, y0=theta[-2:], t=data.get_column_data(names[0]), args=(theta,))
    #     result_py = odeint(func=self.rhs, y0=[11.85718184,  5.9927679 ], t=data.get_column_data(names[0]), args=(theta,))#
    #     return result_py

    # @njit
    # def rhs(self, X, t, theta):
    #     # unpack parameters
    #     x, y = X
    #     alpha, beta, gamma, delta, xt0, yt0 = theta
    #     # equations
    #     dx_dt = alpha * x - beta * x * y
    #     dy_dt = -gamma * y + delta * x * y
    #     return [dx_dt, dy_dt]

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:

        #ODE SIMULATOR
        ode_system = inputs["system"]

        t_start: float = params["initial_time"]
        t_end: float = params["final_time"]
        t_step: float = params["time_step"]
        method = params.get("method", "RK45")

        sim_system: MCMCSystem = ode_system.create_sim_system()
        self.sim_system = sim_system
        sol = sim_system.simulate(t_start, t_end, t_step=t_step, method=method)

        ode_status: ODEStatus = ODEStatus(success=sol.success, message=sol.message)
        t_df = DataFrame(data=sol.t, columns=["time"])
        y_df = DataFrame(data=sol.y, index=sim_system.state_names()).T
        ysol = ODESimResultTable(data=pandas.concat([t_df, y_df], axis=1))
        ysol.set_ode_status(ode_status)


        # COMPUTE LEAST SQUARE
        data = inputs["data"]
        self.data = data
        #TODO : recuperer le nom des paramètres directement depuis le fichier donné en input
        alpha = sim_system.parameters(0)[0]
        beta = sim_system.parameters(0)[1]
        gamma = sim_system.parameters(0)[2]
        delta = sim_system.parameters(0)[3]
        xt0 = sim_system.initial_state()[0]
        yt0 = sim_system.initial_state()[1]
        theta = [alpha,beta,gamma,delta,xt0,yt0]
        results =least_squares(fun = self.ode_model_residuals, x0=theta, args = (data,ysol))

        # put solution obtained from least-squares in prior
        theta = results.x

        tune = params["tune"]
        draws = params["draws"]
        cores = params["cores"]
        prior_type = params["prior_type"]

        parameter_priors = {"alpha": theta[0], "beta": theta[1], "gamma": theta[2],
                            "delta": theta[3], "xt0": theta[4], "yt0": theta[5]}
        param_vars = {}

        with pm.Model() as model:
            for param, prior_spec in parameter_priors.items():
                if prior_type == 'normal':
                    param_vars[param] = pm.Normal(param, mu=prior_spec, sigma=prior_spec)
                elif prior_type == 'uniform':
                    param_vars[param] = pm.Uniform(param, lower=prior_spec, upper=prior_spec)
                elif prior_type == 'truncatednormal':
                    param_vars[param] = pm.TruncatedNormal(param, mu=prior_spec, sigma=prior_spec, lower=0, initval=prior_spec)
                elif prior_type == 'halfnormal':
                    param_vars[param] = pm.HalfNormal(param, sigma=prior_spec)
                elif prior_type == 'halfcauchy':
                    param_vars[param] = pm.HalfCauchy(param, beta=prior_spec)
                else:
                    raise ValueError(f"Invalid prior distribution type for parameter {param}")

            sigma = pm.TruncatedNormal("sigma", mu=10, sigma=10, lower=0)

            # compute pytensor
            theta = pm.math.stack([param_vars["alpha"], param_vars["beta"], param_vars["gamma"], param_vars["delta"], param_vars["xt0"], param_vars["yt0"]])
            self.theta = theta
            #ode_solution = self.pytensor_forward_model_matrix(theta)
            #ode_solution = sim_system.pytorch_forward_model_matrix(theta)
            ode_solution = sim_system.tensorflow_forward_model_matrix(theta)

            # Likelihood
            names = data.column_names
            Y_obs= pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=[data.get_column_data(names[2]), data.get_column_data(names[1])])

            vars_list = list(model.values_to_rvs.keys())[:-1]
            # Inference!
            sampler = "Slice Sampler"
            trace_slice = pm.sample(step=[pm.Slice(vars_list)], tune=tune, draws=draws, cores=cores)

        return {"result": ysol}
