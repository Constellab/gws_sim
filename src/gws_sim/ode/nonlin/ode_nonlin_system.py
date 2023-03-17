# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

# -> globaly import math ode equations
import math as math  # /!\ do not remove
import re
from typing import Any, List, Union

import numpy as np
from gws_core import BadRequestException, Table, resource_decorator
from pandas import DataFrame

# -> globaly import mass action laws for ode equations
from ...law.law import Law as law  # /!\ do not remove
from ..base.ode_base_system import BaseODESystem


@resource_decorator("NonlinarODESystem", human_name="Nonlinear ODE system",
                    short_description="System of nonlinear ordinary differential equations")
class NonlinarODESystem(BaseODESystem):
    """
    NonlinarODESystem class

    System of nonlinear ordinary differential equations `dx/dt = H(x(t), u(t))` where:
      - `t` is time
      - `x(t)` is the n-by-1 state vector, where `n` is the number of state variables
      - `u(t)` is the m-by-1 input vector, where `m` is the number of input variables
      - `H = [h_1(x(t),u(t)), ..., h_n(x(t),u(t))]^T ` is the n-by-1 matrix of functions `h_i(x(t),u(t))`
    """

    DEFAULT_PARAMETER_TABLE_NAME = "Default parameters"
    EQUATION_TABLE_NAME = "Equation table"

    _parameters: Any = None
    _mem_eqn_code: str = None

    _mem_var_names: List[str] = None
    _mem_diff_var_names: List[str] = None
    _mem_diff_eqn_rhs: List[str] = None

    def __init__(self, equations: Union[str, Table, DataFrame] = None, parameters: Union[Table, DataFrame] = None):
        super().__init__()
        if equations is not None:
            self._set_system(equations)

        self._prepare_mem_eqn()  # /!\ do not remove

        if parameters is not None:
            self.set_parameters(parameters)

    def initialize(self):
        self._prepare_mem_data()

    def derivative(self, t: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """ The derivative of the ODE system """
        dxdt: np.ndarray = np.zeros(shape=self.size)
        ctx = {"t": t, "x": x, "u": u, "dxdt": dxdt}
        exec(self._mem_eqn_code, globals(), ctx)
        return dxdt

    def get_parameters(self, as_dict=False):
        """ Get the parameters """
        if self._parameters is not None:
            params = self._parameters
        else:
            params = self.get_default_parameters()
        return params

    def get_default_parameters(self):
        """ Get the default parameters """
        if self.resource_exists(self.DEFAULT_PARAMETER_TABLE_NAME):
            return self.get_resource(self.DEFAULT_PARAMETER_TABLE_NAME)
        else:
            return None

    def get_equations(self) -> Table:
        """ The equation table """
        if not self.resource_exists(self.EQUATION_TABLE_NAME):
            return None
        return self.get_resource(self.EQUATION_TABLE_NAME)

    def get_initial_state(self, as_float=False) -> Union[Table, List[float]]:
        """ The intial state as table or list or float """
        if self._initial_state is not None:
            initial_state = self._initial_state
        else:
            initial_state = self.get_default_initial_state()

        if as_float:
            return self._convert_initial_state_to_float(initial_state)
        else:
            return initial_state

    def get_default_initial_state(self):
        """ Get the default initial state """
        if self.resource_exists(self.DEFAULT_INITIAL_STATE_TABLE_NAME):
            return self.get_resource(self.DEFAULT_INITIAL_STATE_TABLE_NAME)
        else:
            return None

    def _convert_initial_state_to_float(self, initial_state):
        if self._mem_var_names is None:
            raise BadRequestException("No state variables found. The equation table must be set")

        if initial_state is None:
            raise BadRequestException("No initial state defined")

        # evaluate the code to get the initial state as a vector of float
        initial_state_code = initial_state.get_data().iloc[:, 0].tolist()
        initial_state_code = "\n".join(initial_state_code)
        ctx = {}
        try:
            exec(initial_state_code, globals(), ctx)
            initial_state = [ctx[v] for v in self._mem_var_names]
        except Exception:
            raise BadRequestException("An error occured when evaluating the initial state")

        return initial_state

    def _prepare_mem_eqn(self):
        equations = self.get_equations()
        if equations is None:
            return

        if self._mem_diff_eqn_rhs is None:
            self._prepare_mem_eqn()

        self._mem_var_names = []
        self._mem_diff_eqn_rhs = []
        self._mem_diff_var_names = []
        eqn = self.get_equations().get_data().iloc[:, 0].tolist()
        for elt in eqn:
            tab = re.findall(r'\s*d(.+)/dt\s*=\s*(.+)', elt)
            v = tab[0][0]
            self._mem_var_names.append(v)
            self._mem_diff_var_names.append(f"d{v}dt")
            self._mem_diff_eqn_rhs.append(tab[0][1])

    def _prepare_mem_data(self):
        """ Prepare memory data to optimize ODE integration """

        equations = self.get_equations()
        if equations is None:
            return

        self._mem_eqn_code = []
        for i, diff in enumerate(self._mem_diff_eqn_rhs):
            self._mem_eqn_code.append(self._mem_diff_var_names[i] + " = " + diff)

        var_code = ",".join(self._mem_var_names) + ", = x"
        ret_code = "\n".join([f"dxdt[{i}] = {v}" for i, v in enumerate(self._mem_diff_var_names)])
        self._mem_eqn_code = var_code + "\n" + "\n".join(self._mem_eqn_code) + "\n" + ret_code

        parameters = self.get_parameters()
        if parameters is not None:
            param_code = "\n".join(parameters.get_data().iloc[:, 0].tolist())
            self._mem_eqn_code = param_code + "\n" + self._mem_eqn_code

    @ property
    def size(self) -> int:
        """ The size if the ODE system """
        return self.get_equations().shape[0]

    def _set_system(self, equations: Union[str, Table, DataFrame]):
        """ Set the system """
        if isinstance(equations, str):
            equations = equations.split("\n")
            var_ = []
            for elt in equations:
                tab = re.findall(r'\s*d(.+)/dt\s*=\s*(.+)', elt)
                var_.append(tab[0][0])
            equations = DataFrame(equations, index=var_, columns=["equation"])
        elif isinstance(equations, DataFrame):
            equations = Table(equations)
        if not isinstance(equations, Table):
            raise BadRequestException("The equations table must be a DataFrame or a Table")
        equations.name = self.EQUATION_TABLE_NAME
        self.add_resource(equations)

    # -- S --

    def set_parameters(self, parameters: Union[str, Table, DataFrame]):
        parameters = self._format_parameters(parameters)
        self._parameters = parameters

    def set_default_parameters(self, parameters: Union[str, Table, DataFrame]):
        """ Set the parameters """
        parameters = self._format_parameters(parameters)
        parameters.name = self.DEFAULT_PARAMETER_TABLE_NAME
        self.add_resource(parameters)

    def set_initial_state(self, initial_state: Union[str, list, Table, DataFrame]):
        """ Set the initial state """
        initial_state = self._format_initial_state(initial_state)
        self._initial_state = initial_state

    def set_default_initial_state(self, initial_state: Union[str, list, Table, DataFrame]):
        """ Set the default initial state """
        initial_state = self._format_initial_state(initial_state)
        initial_state.name = self.DEFAULT_INITIAL_STATE_TABLE_NAME
        self.add_resource(initial_state)
        # check data size
        x_0 = self._convert_initial_state_to_float(initial_state)
        if len(x_0) != self.size:
            raise BadRequestException("The size of the default initial state data match with the size of the system")

    def _format_initial_state(self, initial_state):
        if isinstance(initial_state, str):
            initial_state = initial_state.split("\n")
            initial_state = Table(DataFrame({"initial_state": initial_state}))
        elif isinstance(initial_state, list):
            initial_state = Table(DataFrame({"initial_state": initial_state}))
        elif isinstance(initial_state, DataFrame):
            initial_state = Table(initial_state)
        if not isinstance(initial_state, Table):
            raise BadRequestException("The initial state table must be a DataFrame or a Table")
        return initial_state

    def _format_parameters(self, parameters):
        if isinstance(parameters, str):
            parameters = parameters.split("\n")
            parameters = Table(DataFrame({"parameters": parameters}))
        elif isinstance(parameters, list):
            parameters = Table(DataFrame({"parameters": parameters}))
        elif isinstance(parameters, DataFrame):
            parameters = Table(parameters)
        if not isinstance(parameters, Table):
            raise BadRequestException("The parameter table must be a DataFrame or a Table")
        return parameters

    @ property
    def state_names(self):
        """ The names of the state variables """
        return self.get_equations().row_names

    def dumps(self) -> dict:
        data = {
            "equations": self._dumps_table_as_dict(self.get_equations(), "equation"),
            "default_initial_state": self._dumps_table_as_list(self.get_default_initial_state(), "initial_state"),
            "default_parameters": self._dumps_table_as_list(self.get_default_parameters(), "parameters"),
        }
        return data

    @classmethod
    def loads(cls, data) -> "NonlinarODESystem":
        equations = cls._load_table_from_dict(data["equations"], "equation")
        default_initial_state = cls._load_table_from_list(data["default_initial_state"], "initial_state")
        default_parameters = cls._load_table_from_list(data["default_parameters"], "parameters")
        nonlin_system = NonlinarODESystem(equations=equations)
        if default_parameters:
            nonlin_system.set_default_parameters(default_parameters)
        if default_initial_state:
            nonlin_system.set_default_initial_state(default_initial_state)
        return nonlin_system

    @staticmethod
    def _dumps_table_as_dict(table, column_name):
        if table is None:
            data = {}
        else:
            data = table.get_data().to_dict(orient="index")
            data = {name: eqn[column_name] for name, eqn in data.items()}
        return data

    @staticmethod
    def _dumps_table_as_list(table, column_name):
        if table is None:
            data = []
        else:
            data = table.get_data().loc[:, column_name].to_list()
        return data

    @staticmethod
    def _load_table_from_dict(data, column_name):
        df = DataFrame.from_dict(data, orient='index')
        if df.shape[0] > 0 and df.shape[1] > 0:
            df.columns = [column_name]
            return Table(df)
        else:
            return None

    @staticmethod
    def _load_table_from_list(data, column_name):
        df = DataFrame({column_name: data})
        if df.shape[0] > 0 and df.shape[1] > 0:
            df.columns = [column_name]
            return Table(df)
        else:
            return None
