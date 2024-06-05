
from abc import abstractmethod
from typing import List, Union

import numpy as np
from scipy.integrate import solve_ivp, odeint, OdeSolution

from gws_core import (BadRequestException, MambaShellProxy)
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

    def before_simulate(self, args):
        """
            Called before simulate
            To override if required
        """

    @abstractmethod
    def initial_state(self, args=None) -> np.ndarray:
        """ The initial state of the system """

    @abstractmethod
    def parameters(self, args=None) -> dict:
        """ The derivative of the system """

    @abstractmethod
    def state_names(self) -> List[str]:
        """ The state names """

    @abstractmethod
    def derivative(self, args=None) -> List[str]:
        """ The derivative of the system """

    def initial_state_(self, args=None) -> np.ndarray:
        return self.initial_state()

    def simulate(self, t_start, t_end, initial_state=None, parameters=None, dataframe: DataFrame = None, args=None) -> Union[PINNSolution, np.ndarray]:

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

        self._cache = {
            "y0": initial_state,
            "t_start": t_start,
            "t_end": t_end,
        }

        def save_data_to_temp_directory(_dataframe, _string_list_equations, _list_params, _initial_state):
            # Create temp dir
            temp_dir = tempfile.mkdtemp()

            # Save dataframe in temp csv file
            csv_file_path = os.path.join(temp_dir, 'dataframe.csv')
            _dataframe.to_csv(csv_file_path, index=False)

            # Save string list in temp txt file for equations
            txt_file_path_equations = os.path.join(
                temp_dir, 'string_list_equations.txt')
            with open(txt_file_path_equations, 'w') as f:
                for item in _string_list_equations:
                    f.write("%s\n" % item)

            # Save string list in temp txt file for params
            txt_file_path_params = os.path.join(
                temp_dir, 'string_list_params.txt')
            with open(txt_file_path_params, 'w') as f:
                for item in _list_params:
                    f.write("%s\n" % str(item))

            txt_file_path_initial_state = os.path.join(
                temp_dir, 'string_list_initial_state.txt')
            with open(txt_file_path_initial_state, 'w') as f:
                for item in _initial_state:
                    f.write("%s\n" % str(item))

            return csv_file_path, txt_file_path_equations, txt_file_path_params, txt_file_path_initial_state, temp_dir

        csv_file_path, txt_file_path_equations, txt_file_path_params, txt_file_path_initial_state, temp_dir = save_data_to_temp_directory(
            dataframe, self.derivative(), self.parameters(), initial_state)

        # Unique name of the virtual env
        env_dir_name = "PinnSystemShellProxy"

        current_path = os.path.abspath(os.path.dirname(__file__))

        # Path of the virtual env file relative to this python file
        env_file_path = os.path.join(current_path,  "../pinn_mamba_env.yml")

        path_script_pinn = os.path.join(current_path, "../_pinn_code.py")

        cmd = f"python {path_script_pinn} {csv_file_path} {txt_file_path_equations} {txt_file_path_params} {t_start} {t_end} {txt_file_path_initial_state}"

        proxy = MambaShellProxy(env_dir_name, env_file_path)

        result = proxy.run(cmd=cmd, shell_mode=True)

        if result != 0: raise Exception("An error occured during the execution of the script.")

        shutil.rmtree(temp_dir)
        y_df = pd.read_csv('../pinn_result.csv')
        if y_df is None:
            return PINNSolution(None, False, 'Error during process')
        os.remove('../pinn_result.csv')
        return PINNSolution(y_df, True, 'Pinn system worked')
