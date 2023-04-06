
from gws_core import BaseTestCase
from gws_sim import SimpleODESystem, ODESimSystem


class TestODESystem(BaseTestCase):

    def test_ode_lorentz(self):

        ode_sys = SimpleODESystem(
            equations=["du/dt = -sigma*(u - v)", "dv/dt = rho*u - v - u*w", "dw/dt = -beta*w + u*v"],
            parameters=["sigma, beta, rho = 10, 2.667, 28"],
            initial_state=["u, v, w = 0, 1, 1.05"]
        )

        code = ode_sys.generate_code()
        print(code)
        sim_sys = ode_sys.create_sim_system()
        self.assertTrue(isinstance(sim_sys, ODESimSystem))

        sol = sim_sys.simulate(t_start=0, t_end=100, t_step=0.1)
        self.assertEqual(sol.success, True)
        self.assertEqual(len(sol.t), 1001)
        print(sol)

    def test_ode_lorentz_from_text(self):
        litteral_text = """
#parameters
sigma = 10
beta = 2.667
rho = 28

#initial_state
u = 0
v, w = 1.0, 1.05

#equations
du/dt = -sigma*(u - v)
dv/dt = rho*u - v - u*w
dw/dt = -beta*w + u*v
"""
        ode_sys = SimpleODESystem.from_text(litteral_text)
        code = ode_sys.generate_code()
        print(code)

        sim_sys = ode_sys.create_sim_system()
        self.assertTrue(isinstance(sim_sys, ODESimSystem))

        sol = sim_sys.simulate(t_start=0, t_end=100, t_step=0.1)
        self.assertEqual(sol.success, True)
        self.assertEqual(len(sol.t), 1001)
        print(sol)
