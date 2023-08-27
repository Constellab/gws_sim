# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from typing import Type

from gws_core import (ConfigParams, ConfigSpecs, File,
                      FileHelper, ResourceExporter, StrParam,
                      exporter_decorator)

from .mc_system import MCSystem


@exporter_decorator(unique_name="MCSystemExporter", human_name="MC system exporter",
                    source_type=MCSystem, target_type=File)
class MCSystemExporter(ResourceExporter):
    ALLOWED_FILE_FORMATS = ["py"]
    DEFAULT_FILE_FORMAT = "py"
    config_specs: ConfigSpecs = {
        'file_name': StrParam(default_value="system", short_description="File name (without extension)"),
        'file_format':
        StrParam(
            allowed_values=ALLOWED_FILE_FORMATS,
            default_value=DEFAULT_FILE_FORMAT,
            short_description="File format")}

    def export_to_path(
            self, source: MCSystem, dest_dir: str, params: ConfigParams, target_type: Type[File]) -> File:
        """
        Export an ODE system

        :param file_path: The destination file path
        :type file_path: str
        """

        file_name = params.get_value("file_name", source.name or "system")
        file_format = FileHelper.clean_extension(params.get_value("file_format", "py"))
        file_path = os.path.join(dest_dir, file_name + '.' + file_format)

        with open(file_path, 'w', encoding="utf-8") as fp:
            data = source.dumps(text=True)
            fp.write(data)

        return target_type(path=file_path)
