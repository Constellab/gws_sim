# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Type
import json
from gws_core import (ConfigParams, ConfigSpecs, File, ResourceImporter,
                      importer_decorator, StrParam)

from .mc_prior_dict import MCPriorDict


@importer_decorator("MCPriorDictImporter", human_name="Monte-Carlo priors importer", source_type=File,
                    target_type=MCPriorDict, supported_extensions=["json"])
class MCPriorDictImporter(ResourceImporter):
    """ MCSystemImporter """
    ALLOWED_FILE_FORMATS = ["json"]
    DEFAULT_FILE_FORMAT = "json"
    config_specs: ConfigSpecs = {
        'file_format':
        StrParam(
            allowed_values=ALLOWED_FILE_FORMATS,
            default_value=DEFAULT_FILE_FORMAT,
            short_description="File format")}

    def import_from_path(
            self, source: File, params: ConfigParams, target_type: Type[MCPriorDict]) -> MCPriorDict:
        """
        Import an ODE system
        """

        with open(source.path, 'r', encoding="utf-8") as fp:
            data: str = json.load(fp)

        return target_type(data=data)
