# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from typing import Type
import json
from gws_core import (ConfigParams, ConfigSpecs, File,
                      FileHelper, ResourceExporter, StrParam,
                      exporter_decorator)

from .mc_prior_dict import MCPriorDict


@exporter_decorator(unique_name="MCPriorDictExporter", human_name="Monte-Carlo priors exporter",
                    source_type=MCPriorDict, target_type=File)
class MCPriorDictExporter(ResourceExporter):
    ALLOWED_FILE_FORMATS = ["json"]
    DEFAULT_FILE_FORMAT = "json"
    config_specs: ConfigSpecs = {
        'file_name': StrParam(default_value="system", short_description="File name (without extension)"),
        'file_format':
        StrParam(
            allowed_values=ALLOWED_FILE_FORMATS,
            default_value=DEFAULT_FILE_FORMAT,
            short_description="File format")}

    def export_to_path(
            self, source: MCPriorDict, dest_dir: str, params: ConfigParams, target_type: Type[File]) -> File:
        """
        Export a Monte-Carlo dictionnary of priors

        :param file_path: The destination file path
        :type file_path: str
        """

        file_name = params.get_value("file_name", source.name or "mc_priors")
        file_format = FileHelper.clean_extension(params.get_value("file_format", "json"))
        file_path = os.path.join(dest_dir, file_name + '.' + file_format)

        with open(file_path, 'w', encoding="utf-8") as fp:
            json.dump(source.get_data(), fp)

        return target_type(path=file_path)
