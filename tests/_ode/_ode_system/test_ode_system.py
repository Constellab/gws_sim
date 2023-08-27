
from gws_core import BaseTestCase
from gws_sim import ODESystem, ODESystemHelper


class TestODESystem(BaseTestCase):

    def test_ode_lorentz(self):

        pycode = """
from gws_sim import ODESystemHelper
class System(ODESystemHelper):
    def initial_state(self, args=None):
        return [0, 1, 1.05]

    def parameters(self, t, args=None):
        return 10, 2.667, 28

    def derivative(self, t, x, args=None):
        u, v, w = x
        sigma, rho, beta = self.parameters(t, args)

        dudt = -sigma*(u - v)
        dvdt = rho*u - v - u*w
        dwdt = -beta*w + u*v
        return [dudt, dvdt, dwdt]
"""
        ode_sys = ODESystem(
            code=pycode
        )

        sim_sys = ode_sys.create_sim_system_helper()
        self.assertTrue(isinstance(sim_sys, ODESystemHelper))

        sol = sim_sys.simulate(t_start=0, t_end=100, t_step=0.1)
        self.assertEqual(sol.success, True)
        self.assertEqual(len(sol.t), 1001)
        print(sol)
