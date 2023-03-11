# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import List, Union

import numpy as np
from gws_core import BadRequestException, Table, resource_decorator
from pandas import DataFrame

from ..base.ode_base_system import BaseODESystem


@resource_decorator("LinarODESystem", human_name="Linear ODE system",
                    short_description="System of linear ordinary differential equations")
class LinarODESystem(BaseODESystem):
    """
    LinarODESystem class

    System of linear ordinary differential equations `dx/dt = A x(t) + B u(t)` where:
      - `t` is time
      - `x(t)` is the n-by-1 state vector, where `n` is the number of state variables
      - `u(t)` is the m-by-1 input vector, where `m` is the number of input variables
      - `A` is the n-by-n state matrix
      - `B` is the n-by-m input matrix
    """

    STATE_MATRIX_NAME = "State matrix A"
    INPUT_MATRIX_NAME = "Input matrix B"
    OUTPUT_MATRIX_NAME = "Output matrix C"

    _mem_A: np.ndarray = None
    _mem_B: np.ndarray = None
    _mem_C: np.ndarray = None
    _initial_state = None

    def __init__(self, A: Union[Table, DataFrame] = None, B: Union[Table, DataFrame] = None):
        super().__init__()
        if A is not None:
            self._set_system(A, B)

    def initialize(self):
        self._prepare_mem_data()

    def derivative(self, t: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        The derivative of the ODE system
            - x is the value of the state
            - u is the value of the input
        """

        if self._mem_B is None:
            return self._mem_A @ x
        else:
            return self._mem_A @ x + self._mem_B @ u

    @property
    def size(self) -> int:
        """ The size if the state vector of the ODE system """
        return self.A.shape[0]

    @property
    def A(self) -> Table:
        """ The state matrix """
        return self.get_resource(self.STATE_MATRIX_NAME)

    @property
    def B(self) -> Table:
        """ The input matrix """
        if self.resource_exists(self.INPUT_MATRIX_NAME):
            return self.get_resource(self.INPUT_MATRIX_NAME)
        else:
            return None

    @property
    def C(self) -> Table:
        """ The output matrix """
        raise BadRequestException("Not yet implemented")

    def _prepare_mem_data(self):
        """ Prepare memory data to optimize ODE integration """
        self._mem_A = self.A.get_data().to_numpy()
        if self.B is not None:
            self._mem_B = self.B.get_data().to_numpy()

    def _set_system(self, A: Union[Table, DataFrame], B: Union[Table, DataFrame] = None):
        """ Set the system """
        if isinstance(A, DataFrame):
            A = Table(A)
        if isinstance(B, DataFrame):
            B = Table(B)

        if not isinstance(A, Table):
            raise BadRequestException("The state matrix A must be a Table or DataFrame")

        if A.shape[0] != A.shape[1]:
            raise BadRequestException("The state matrix A have same number of rows and columns")

        A.name = self.STATE_MATRIX_NAME
        self.add_resource(A)

        if B is not None:
            if not isinstance(B, Table):
                raise BadRequestException("The input matrix B must be a Table or DataFrame")
            if A.shape[0] != B.shape[1]:
                raise BadRequestException("The number of row of A and B must be the same")

            B.name = self.INPUT_MATRIX_NAME
            self.add_resource(B)

    def get_initial_state(self, as_float=True):
        """ Get the initial state """
        if self._initial_state is not None:
            initial_state = self._initial_state
        else:
            initial_state = self.get_default_initial_state()

        if as_float:
            return initial_state.get_data().iloc[:, 0].tolist()
        else:
            return initial_state

    def get_default_initial_state(self):
        """ Get the default initial state """
        if self.resource_exists(self.DEFAULT_INITIAL_STATE_TABLE_NAME):
            return self.get_resource(self.DEFAULT_INITIAL_STATE_TABLE_NAME)
        else:
            return None

    def set_initial_state(self, initial_state: Union[Table, DataFrame]):
        """ Set the default initial state """
        initial_state = self._prepare_initial_state(initial_state)
        self._initial_state = initial_state

    def set_default_initial_state(self, initial_state: Union[Table, DataFrame]):
        """ Set the default initial state """
        initial_state = self._prepare_initial_state(initial_state)
        initial_state.name = self.DEFAULT_PARAMETER_TABLE_NAME
        self.add_resource(initial_state)

    def _prepare_initial_state(self, initial_state):
        if isinstance(initial_state, list):
            df = DataFrame({"initial_state": initial_state})
            initial_state = Table(df)
        if isinstance(initial_state, DataFrame):
            initial_state = Table(initial_state)
        if not isinstance(initial_state, Table):
            raise BadRequestException("The initial state table must be a DataFrame or a Table")
        return initial_state

    @property
    def state_names(self) -> list:
        """ The names of the state variables """
        return self.A.row_names
