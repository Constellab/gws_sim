# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
import tempfile

from gws_core import (resource_decorator, BadRequestException, Text, PackageHelper, ResourceSet)
from ..helper.mc_system_helper import MCSystemHelper


@resource_decorator("MCSystem", human_name="MC system",
                    short_description="Monte-Carlo system", hide=True)
class MCSystem(ResourceSet):

    SYSTEM_CODE_NAME = "MC system code"

    def __init__(self, code=None):
        super().__init__()
        if code is not None:
            self._set_code(code)

    # -- C --

    def create_sim_system_helper(self) -> MCSystemHelper:
        """ Creates a MC system """
        code = self.get_code()
        if code is None:
            raise BadRequestException("No code defined")

        _, snippet_filepath = tempfile.mkstemp(suffix=".py")
        with open(snippet_filepath, 'w', encoding="utf-8") as fp:
            fp.write(code)

        module = PackageHelper.load_module_from_file(snippet_filepath)

        try:
            os.unlink(snippet_filepath)
        except:
            pass

        return module.System()

    # -- G --

    def get_code(self):
        if not self.resource_exists(self.SYSTEM_CODE_NAME):
            return None
        return self.get_resource(self.SYSTEM_CODE_NAME).get_data()

    # -- D --

    def dumps(self) -> str:
        """ Dump the system to a destination data """

        return self.get_code()

    # -- L --

    @classmethod
    def loads(cls, data: str):
        """ Load from a source data """

        return MCSystem(code=data)

    # -- S --

    def _set_code(self, code: str):
        """ Set the code """
        if isinstance(code, str):
            text = Text(data=code)
        else:
            raise BadRequestException("The code must be a raw text")
        text.name = self.SYSTEM_CODE_NAME
        self.add_resource(text)
