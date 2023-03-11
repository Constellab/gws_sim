# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import json
import os
from typing import Type

from gws_core import (BadRequestException, ConfigParams, ConfigSpecs, File,
                      FileHelper, ResourceExporter, StrParam,
                      exporter_decorator)
from pandas import DataFrame

from ..ode_nonlin_system import NonlinarODESystem


@exporter_decorator(unique_name="NonlinarODESystemExporter", human_name="Nonlinar ODE system exporter",
                    source_type=NonlinarODESystem, target_type=File)
class NetworkExporter(ResourceExporter):
    ALLOWED_FILE_FORMATS = ["json"]
    DEFAULT_FILE_FORMAT = "json"
    config_specs: ConfigSpecs = {
        'file_name': StrParam(default_value="network", short_description="File name (without extension)"),
        'file_format':
        StrParam(
            allowed_values=ALLOWED_FILE_FORMATS,
            default_value=DEFAULT_FILE_FORMAT,
            short_description="File format")}

    async def export_to_path(self, source: NonlinarODESystem, dest_dir: str, params: ConfigParams, target_type: Type[File]) -> File:
        """
        Export an ODE system

        :param file_path: The destination file path
        :type file_path: str
        """

        file_name = params.get_value("file_name", source.name or "system")
        file_format = FileHelper.clean_extension(params.get_value("file_format", "json"))
        file_path = os.path.join(dest_dir, file_name + '.' + file_format)

        with open(file_path, 'w', encoding="utf-8") as fp:
            data = source.dumps()
            json.dump(data, fp)

        return target_type(path=file_path)
