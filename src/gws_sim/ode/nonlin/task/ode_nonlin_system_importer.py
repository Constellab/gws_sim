# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import json
from typing import Type

from gws_core import (ConfigParams, ConfigSpecs, File, ResourceImporter,
                      importer_decorator)

from ..ode_nonlin_system import NonlinarODESystem


@importer_decorator("NonlinarODESystemImporter", human_name="Nonlinear ODE system importer", source_type=File,
                    target_type=NonlinarODESystem, supported_extensions=["json"])
class NonlinarODESystemImporter(ResourceImporter):
    """ NonlinarODESystemImporter """
    config_specs: ConfigSpecs = {
    }

    def import_from_path(
            self, source: File, params: ConfigParams, target_type: Type[NonlinarODESystem]) -> NonlinarODESystem:
        """
        Import an ODE system
        """

        with open(source.path, 'r', encoding="utf-8") as fp:
            data = json.load(fp)

        nonlin_system = target_type.loads(data)
        return nonlin_system
