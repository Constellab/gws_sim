
from abc import abstractmethod
from typing import List, Union

import numpy as np
from scipy.integrate import solve_ivp, odeint, OdeSolution

from gws_core import (BadRequestException, MambaShellProxy, MessageDispatcher)
from ...helper.base_sim_system_helper import BaseSimSystemHelper
from pandas import DataFrame
import pandas as pd
import os
import tempfile
import shutil


class PINNSolution:
    y = None
    success = None
    message = None

    def __init__(self, y, success, message):
        self.y = y
        self.success = success
        self.message = message


class PINNSystemHelper(BaseSimSystemHelper):

    _cache: dict = None

    _message_dispatcher: MessageDispatcher = None

    def before_simulate(self, args):
        """
            Called before simulate
            To override if required
        """

    @abstractmethod
    def initial_state(self, args=None) -> np.ndarray:
        """ The initial state of the system """

    @abstractmethod
    def parameters(self, args=None) -> np.ndarray:
        """ The derivative of the system """

    @abstractmethod
    def additional_functions(self, args=None) -> str:
        """ additional_functions """

    @abstractmethod
    def state_names(self) -> List[str]:
        """ The state names """

    @abstractmethod
    def derivative(self, args=None) -> np.ndarray:
        """ The derivative of the system """

    def initial_state_(self, args=None) -> np.ndarray:
        return self.initial_state()

    def simulate(self, t_start: float, t_end: float, number_hidden_layers:int, width_hidden_layers:int, number_iterations:int, number_iterations_predictive_controller:int, control_horizon:float, simulator_type:str, initial_state=None, parameters=None, dataframe: DataFrame = None,
                additional_functions=None, args=None) -> Union[PINNSolution, np.ndarray]:

        if t_end <= t_start:
            raise BadRequestException(
                "The final time must be greater than the initial time")

        if dataframe is None:
            raise BadRequestException(
                "The dataframe is None"
            )

        self.before_simulate(args)
        if parameters is None:
            parameters = self.parameters(args=args)
        if args is None:
            args = parameters
        else:
            args = [parameters, args]

        if initial_state is None:
            initial_state = self.initial_state(args)

        if additional_functions is None:
            additional_functions = self.additional_functions(args)

        self._cache = {
            "y0": initial_state,
            "t_start": t_start,
            "t_end": t_end,
        }

        csv_file_path, txt_file_path_equations, txt_file_path_params, txt_file_path_initial_state, txt_file_path_additional_functions, temp_dir = self.save_data_to_temp_directory(dataframe, initial_state)

        # Unique name of the virtual env
        env_dir_name = "PinnSystemShellProxy"

        current_path = os.path.abspath(os.path.dirname(__file__))

        # Path of the virtual env file relative to this python file
        env_file_path = os.path.join(current_path,  "../pinn_mamba_env.yml")

        path_script_pinn = os.path.join(current_path, "../_pinn_code.py")

        cmd = f"python3 '{path_script_pinn}' '{csv_file_path}' '{txt_file_path_equations}' '{txt_file_path_params}' '{t_start}' '{t_end}' '{txt_file_path_initial_state}' '{number_hidden_layers}' '{width_hidden_layers}' '{number_iterations}' '{txt_file_path_additional_functions}' '{number_iterations_predictive_controller}' '{control_horizon}' '{simulator_type}'"

        proxy = MambaShellProxy(
            env_dir_name, env_file_path, None, self._message_dispatcher)

        result = proxy.run(cmd=cmd, shell_mode=True)

        if result != 0:
            raise Exception(
                "An error occured during the execution of the script.")

        shutil.rmtree(temp_dir)
        y_df = pd.read_csv('../pinn_result.csv')
        if y_df is None:
            return PINNSolution(None, False, 'Error during process')
        os.remove('../pinn_result.csv')
        return PINNSolution(y_df, True, 'Pinn system worked')

    def save_data_to_temp_directory(self, dataframe, initial_state):
        # Create temp dir
        temp_dir = tempfile.mkdtemp()

        # Save dataframe in temp csv file
        csv_file_path = os.path.join(temp_dir, 'dataframe.csv')
        dataframe.to_csv(csv_file_path, index=False)

        # Save string list in temp txt file for equations
        txt_file_path_equations = os.path.join(
            temp_dir, 'string_list_equations.txt')
        with open(txt_file_path_equations, 'w') as f:
            string_list_equations = self.derivative()
            for item in string_list_equations:
                f.write("%s\n" % str(item))

        # Save string list in temp txt file for params
        txt_file_path_params = os.path.join(
            temp_dir, 'string_list_params.txt')
        with open(txt_file_path_params, 'w') as f:
            list_params = self.parameters()
            for item in list_params:
                f.write("%s\n" % str(item))

        # Save string list in temp txt file for additional functions
        txt_file_path_additional_functions = os.path.join(
            temp_dir, 'string_additional_functions.txt')
        with open(txt_file_path_additional_functions, 'w') as f:
            string_additional_functions = self.additional_functions()
            f.write(string_additional_functions)

        txt_file_path_initial_state = os.path.join(
            temp_dir, 'string_list_initial_state.txt')
        with open(txt_file_path_initial_state, 'w') as f:
            for item in initial_state:
                f.write("%s\n" % str(item))

        return csv_file_path, txt_file_path_equations, txt_file_path_params, txt_file_path_initial_state, txt_file_path_additional_functions, temp_dir

    def set_message_dispatcher(self, message_dispatcher: MessageDispatcher) -> None:
        self._message_dispatcher = message_dispatcher
