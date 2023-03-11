
# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import Table, resource_decorator


@resource_decorator("ODEStateTable", human_name="ODE state table", short_description="Table of state")
class ODEStateTable(Table):
    """
    ODEStateTable class

    Table representing the vector of the state of a system of ordinary differential equations.

    * The first column contains the names of the state variables
    * The following column contains the values of the state variables at corresponding time points

    For example:

    | time      | x1  | x2  | ... | xn   |
    |-----------|-----|-----|-----|------|
    | 0         | 0.0 | 2.1 | ... | 0.1  |
    | 1         | 1.2 | 2.0 | ... | 10.0 |
    | ...       | ... | ... | ... | ...  |
    | 100       | 3.3 | 5.3 | ... | 0.2  |
    """
