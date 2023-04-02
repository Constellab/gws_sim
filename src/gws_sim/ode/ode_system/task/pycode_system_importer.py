# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Type

from gws_core import (ConfigParams, ConfigSpecs, File, ResourceImporter,
                      importer_decorator, StrParam)

from ..pycode_ode_system import PyCodeODESystem


@importer_decorator("PyCodeODESystemImporter", human_name="PyCode ODE system importer", source_type=File,
                    target_type=PyCodeODESystem, supported_extensions=["py", "txt"])
class PyCodeODESystemImporter(ResourceImporter):
    """ PyCodeODESystemImporter """
    ALLOWED_FILE_FORMATS = ["py", "txt"]
    DEFAULT_FILE_FORMAT = "py"
    config_specs: ConfigSpecs = {
        'file_format':
        StrParam(
            allowed_values=ALLOWED_FILE_FORMATS,
            default_value=None,
            short_description="File format")}

    def import_from_path(
            self, source: File, params: ConfigParams, target_type: Type[PyCodeODESystem]) -> PyCodeODESystem:
        """
        Import an ODE system
        """

        with open(source.path, 'r', encoding="utf-8") as fp:
            data: str = fp.read()

        return target_type.loads(data)
