
# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import TypedDict

from gws_core import Table, TechnicalInfo, resource_decorator

ODEStatus = TypedDict("ODEStatus", {
    "success": str,
    "message": str,
})


@resource_decorator("ODESimResultTable")
class ODESimResultTable(Table):

    def set_ode_status(self, ode_status: ODEStatus):
        self.add_technical_info(TechnicalInfo(key='success', value=f'{ode_status["success"]}'))
        self.add_technical_info(TechnicalInfo(key='message', value=f'{ode_status["message"]}'))
