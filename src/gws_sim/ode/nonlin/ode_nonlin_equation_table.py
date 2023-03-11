
# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import Table, resource_decorator


@resource_decorator("ODENonlinEquationTable", human_name="Nonlinear ODE equation table",
                    short_description="Table of nonlinear ordinary differential equations")
class ODENonlinEquationTable(Table):
    pass
