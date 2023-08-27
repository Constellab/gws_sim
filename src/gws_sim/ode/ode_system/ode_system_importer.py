# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Type

from gws_core import (ConfigParams, ConfigSpecs, File, ResourceImporter,
                      importer_decorator, StrParam)

from .ode_system import ODESystem


@importer_decorator("ODESystemImporter", human_name="ODE system importer", source_type=File,
                    target_type=ODESystem, supported_extensions=["py"])
class ODESystemImporter(ResourceImporter):
    """ ODESystemImporter """
    ALLOWED_FILE_FORMATS = ["py"]
    DEFAULT_FILE_FORMAT = "py"
    config_specs: ConfigSpecs = {
        'file_format':
        StrParam(
            allowed_values=ALLOWED_FILE_FORMATS,
            default_value=DEFAULT_FILE_FORMAT,
            short_description="File format")}

    def import_from_path(
            self, source: File, params: ConfigParams, target_type: Type[ODESystem]) -> ODESystem:
        """
        Import an ODE system
        """

        with open(source.path, 'r', encoding="utf-8") as fp:
            data: str = fp.read()

        return target_type.loads(data)
