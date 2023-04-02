# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import json
from typing import Type

from gws_core import (ConfigParams, ConfigSpecs, File, ResourceImporter,
                      importer_decorator, StrParam, FileHelper)

from ..simple_ode_system import SimpleODESystem


@importer_decorator("SimpleODESystemImporter", human_name="Simple ODE system importer", source_type=File,
                    target_type=SimpleODESystem, supported_extensions=["json", "txt"])
class SimpleODESystemImporter(ResourceImporter):
    """ SimpleODESystemImporter """
    ALLOWED_FILE_FORMATS = ["json", "txt"]
    DEFAULT_FILE_FORMAT = "json"
    config_specs: ConfigSpecs = {
        'file_format':
        StrParam(
            allowed_values=ALLOWED_FILE_FORMATS,
            default_value=None,
            short_description="File format")}

    def import_from_path(
            self, source: File, params: ConfigParams, target_type: Type[SimpleODESystem]) -> SimpleODESystem:
        """
        Import an ODE system
        """

        extention = source.extension
        file_format = FileHelper.clean_extension(params.get_value("file_format", extention))

        if file_format == "json":
            with open(source.path, 'r', encoding="utf-8") as fp:
                data: dict = json.load(fp)
        else:
            with open(source.path, 'r', encoding="utf-8") as fp:
                data: str = fp.read()

        return target_type.loads(data)
