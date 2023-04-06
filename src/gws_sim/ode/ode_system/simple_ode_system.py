# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
import re
from typing import Union, List
import tempfile

from pandas import DataFrame

from gws_core import resource_decorator, BadRequestException, Table, PackageHelper, view, TextView
from ..sim_system.ode_sim_system import ODESimSystem
from .base_ode_system import BaseODESystem


@resource_decorator("SimpleODESystem", hide=True)
class SimpleODESystem(BaseODESystem):

    INITIAL_STATE_TABLE_NAME = "Initial state"
    PARAMETER_TABLE_NAME = "Parameters"
    EQUATION_TABLE_NAME = "System"

    def __init__(self, equations=None, parameters=None, initial_state=None):
        super().__init__()
        if equations is not None:
            self._set_equations(equations)
            self._set_parameters(parameters)
            self._set_initial_state(initial_state)

    # -- C --

    def create_sim_system(self) -> ODESimSystem:
        """ Create a  """
        code = self.generate_code()
        _, snippet_filepath = tempfile.mkstemp(suffix=".py")
        with open(snippet_filepath, 'w', encoding="utf-8") as fp:
            fp.write(code)

        module = PackageHelper.load_module_from_file(snippet_filepath)

        try:
            os.unlink(snippet_filepath)
        except:
            pass

        return module.Model()

    # -- D --

    def dumps(self, text=False) -> Union[dict, str]:
        """ Dump the system to a destination data """
        equations = self.get_equations().get_data().iloc[:, 0].to_list()
        initial_state = self.get_initial_state().get_data().iloc[:, 0].to_list()
        parameters = self.get_parameters().get_data().iloc[:, 0].to_list()

        if text is True:
            data = "" +\
                "#parameters\n" + \
                "\n".join(parameters) + "\n\n" +\
                "#initial_state\n" + \
                "\n".join(initial_state) + "\n\n" +\
                "#equations\n" + \
                "\n".join(equations) + "\n"
        else:
            data = {
                "parameters": parameters,
                "initial_state": initial_state,
                "equations": equations,
            }
        return data

    # -- F --

    @ classmethod
    def from_text(cls, text: str) -> 'SimpleODESystem':
        """ Create a SimpleODESystem from a text """
        lines = text.split("\n")
        data = {
            "#parameters": [],
            "#initial_state": [],
            "#equations": []
        }

        current_key = None
        for line in lines:
            line = line.strip("\r\n ")
            if len(line) == 0:
                continue

            for k in data:
                if line == k:
                    current_key = k
                    break
            if current_key is not None:
                data[current_key].append(line)

        return SimpleODESystem(
            parameters=data["#parameters"],
            initial_state=data["#initial_state"],
            equations=data["#equations"]
        )

    # -- G --

    def get_equations(self) -> Table:
        """ The equation table """
        if not self.resource_exists(self.EQUATION_TABLE_NAME):
            return None
        return self.get_resource(self.EQUATION_TABLE_NAME)

    def get_parameters(self):
        """ Get the parameters """
        if self.resource_exists(self.PARAMETER_TABLE_NAME):
            return self.get_resource(self.PARAMETER_TABLE_NAME)
        else:
            return None

    def get_initial_state(self):
        """ Get the initial state """
        if self.resource_exists(self.INITIAL_STATE_TABLE_NAME):
            return self.get_resource(self.INITIAL_STATE_TABLE_NAME)
        else:
            return Table()

    def _generate_equation_code(self, state, params) -> str:
        code_list = []
        dvar_list = []
        lines = self.get_equations().get_data().iloc[:, 0].tolist()
        for line in lines:
            if line.startswith("#"):
                continue
            parts = re.findall(r'\s*d(.+)/dt\s*=\s*(.+)', line)
            var = parts[0][0].strip()
            rhs = parts[0][1].strip()

            code_list.append(f"d{var}dt = {rhs}")
            dvar_list.append(f"d{var}dt")

        code = "" +\
            "   def derivative(self, t, x, args = None):\n" +\
            "      " + f"{state} = x\n" +\
            "      " + f"{params} = self.parameters(t, args)\n\n" +\
            "      " + "\n      ".join(code_list) + "\n" +\
            "      return" + " (" + ",".join(dvar_list) + ")" + "\n"

        return code

    def _generate_parameter_code(self, parameters: str = None) -> str:
        default_parameter_data = self.get_parameters().get_data()
        if default_parameter_data.empty:
            if len(parameters) == 0:
                code = "" +\
                    "   def parameters(self, t, args = None):\n" +\
                    "      pass"
            else:
                raise BadRequestException(
                    "No parameters defined. Please define parameters before.")

        param_tab = {}
        if parameters is not None:
            for line in parameters:
                if line.startswith("#"):
                    continue
                parts = re.findall(r'\s*(.+)\s*=\s*(.+)', line)
                lhs = parts[0][0].strip()
                rhs = parts[0][1].strip()
                param_tab[lhs] = rhs

        code_list = []
        param_list = []

        lines = default_parameter_data.iloc[:, 0].tolist()
        for line in lines:
            if line.startswith("#"):
                continue
            parts = re.findall(r'\s*(.+)\s*=\s*(.+)', line)
            lhs = parts[0][0].strip()
            rhs = parts[0][1].strip()

            if lhs in param_tab:
                # replace the parameter
                code_list.append(f"{lhs} = {param_tab[lhs]}")
            else:
                # use the parameter
                code_list.append(f"{lhs} = {rhs}")

            param_list.append(lhs)

        joined_param_list = ",".join(param_list)
        code = "" +\
            "   def parameters(self, t, args = None):\n" +\
            "      " + "\n      ".join(code_list) + "\n" +\
            "      return" + " (" + joined_param_list + ")" + "\n"

        return joined_param_list, code

    def _generate_initial_state_code(self, initial_state: str = None) -> str:
        state_tab = {}
        if initial_state is not None:
            for line in initial_state:
                if line.startswith("#"):
                    continue
                parts = re.findall(r'\s*(.+)\s*=\s*(.+)', line)
                lhs = parts[0][0].strip()
                rhs = parts[0][1].strip()
                state_tab[lhs] = rhs

        code_list = []
        state_list = []
        lines = self.get_initial_state().get_data().iloc[:, 0].tolist()
        for line in lines:
            if line.startswith("#"):
                continue
            parts = re.findall(r'\s*(.+)\s*=\s*(.+)', line)
            lhs = parts[0][0].strip()
            rhs = parts[0][1].strip()

            if lhs in state_tab:
                # replace the parameter
                code_list.append(f"{lhs} = {state_tab[lhs]}")
            else:
                # use the parameter
                code_list.append(f"{lhs} = {rhs}")

            state_list.append(lhs)

        joined_state_list = ",".join(state_list)
        code = "" +\
            "   def initial_state(self, args = None):\n" +\
            "      " + "\n      ".join(code_list) + "\n" +\
            "      return" + " (" + joined_state_list + ")" + "\n"

        return joined_state_list, code

    def _generate_state_names_code(self, state):
        state = '"' + state.replace(" ", "").replace(",", '","') + '"'
        code = "" +\
            "   def state_names(self):\n" +\
            "      return" + " (" + state + ")" + "\n"

        return code

    def generate_code(self):
        state, initial_state_code = self._generate_initial_state_code()
        state_names_code = self._generate_state_names_code(state)
        params, parameter_code = self._generate_parameter_code()

        code = "" +\
            "import math\n" +\
            "from gws_sim import ODESimSystem, Law\n" +\
            "from gws_sim import Law as law\n\n" +\
            "class Model(ODESimSystem):\n" +\
            state_names_code + "\n" +\
            initial_state_code + "\n" +\
            parameter_code + "\n" +\
            self._generate_equation_code(state, params)

        return code

    # -- L --

    @ classmethod
    def loads(cls, data: Union[dict, str]):
        """ Load from a source data """

        if isinstance(data, dict):
            return SimpleODESystem(
                equations=data["equations"],
                initial_state=data["initial_state"],
                parameters=data["parameters"],
            )
        else:
            return SimpleODESystem.from_text(data)

    # -- P --

    def _parse_text(self, text: str, pattern=None, name=None):
        """ Parse text """
        if isinstance(text, str):
            text = text.split("\n")

        if len(text) == 0:
            return Table()

        lhs = []
        valid_lines = []
        for line in text:
            line = line.strip("\r\n ")
            if len(line) != 0:
                if line.startswith("#"):
                    # TODO: check if we skip comments
                    valid_lines.append(line)
                    lhs.append("")
                else:
                    valid_lines.append(line)
                    if re is not None:
                        parts = re.findall(pattern, line)
                        lhs.append(parts[0][0].strip())

        data = DataFrame(valid_lines, index=lhs, columns=[name])
        return Table(data=data)

    # -- S --

    def _set_equations(self, equations: Union[str, List[str]]):
        """ Set the system """
        if isinstance(equations, (str, list)):
            equations = self._parse_text(equations, pattern=r'\s*d(.+)/dt\s*=\s*(.+)', name="equation")
        else:
            raise BadRequestException("The equations must be a string or list of string")
        equations.name = self.EQUATION_TABLE_NAME
        self.add_resource(equations)

    def _set_parameters(self, parameters: Union[str, List[str]]):
        """ Set the parameters """
        if isinstance(parameters, (str, list)):
            parameters = self._parse_text(parameters, pattern=r'\s*(.+)\s*=\s*(.+)', name="parameter")
        else:
            raise BadRequestException("The parameters must be a string or list of string")
        parameters.name = self.PARAMETER_TABLE_NAME
        self.add_resource(parameters)

    def _set_initial_state(self, initial_state: Union[str, List[str]]):
        """ Set the initial_state """
        if isinstance(initial_state, (str, list)):
            initial_state = self._parse_text(initial_state, pattern=r'\s*(.+)\s*=\s*(.+)', name="initial_state")
            if initial_state.get_data().empty:
                raise BadRequestException("An initial_state is required")
        else:
            raise BadRequestException("The initial_state must be a string or list of string")
        initial_state.name = self.INITIAL_STATE_TABLE_NAME
        self.add_resource(initial_state)

    # -- VIEWS ---

    @view(view_type=TextView, human_name="View pycode", short_description="Python ODE code")
    def view_code(self, _) -> TextView:
        text = self.generate_code()
        return TextView(data=text)
