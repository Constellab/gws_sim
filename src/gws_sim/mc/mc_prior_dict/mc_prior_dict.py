
# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Union

from gws_core import resource_decorator, JSONDict


@resource_decorator("MCPriorDict", human_name="Monte-Carlo priors", hide=True)
class MCPriorDict(JSONDict):
    """ MCPriorDict

    Dictionnary containing the priors of a Monte-Carlo system

    The general structure of a prior dictionnary is as follow:

    ```
        {
            "mu": [
                {"name": "alpha", "type": "TruncatedNormal", "mu": alpha_h, "sigma": 0.1, "lower": 0, "initval": alpha_h},
                {"name": "beta", "type": "TruncatedNormal", "mu": beta_h, "sigma": 0.1, "lower": 0, "initval": beta_h},
                {"name": "gamma", "type": "TruncatedNormal", "mu": gamma_h, "sigma": 0.1, "lower": 0, "initval": gamma_h},
                {"name": "delta", "type": "TruncatedNormal", "mu": delta_h, "sigma": 0.1, "lower": 0, "initval": delta_h},
                {"name": "xto", "type": "TruncatedNormal", "mu": xt0_h, "sigma": 0.1, "lower": 0, "initval": xt0_h},
                {"name": "yto", "type": "TruncatedNormal", "mu": yt0_h, "sigma": 0.1, "lower": 0, "initval": yt0_h},
            ]
            "likelihood": {
                "type": "Normal",
                "sigma": {"name": "sigma", "type": "HalfNormal", "sigma": 10}
            }
        }
    ```

    where:

    - `mu` gives the priors of the parameters and initial states of the ODE system
    - `likelihood` define the distribution of the data and the priors of the distribution parameters

    """

    def check_resource(self) -> Union[str, None]:

        data = self.get_data()
        if "mu" not in data:
            return "The key 'mu' is not found in the prior dictionnary."
        if "likelihood" not in data:
            return "The key 'likelihood' is not found in the prior dictionnary."

        return None
