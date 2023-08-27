
from gws_core import BaseTestCase, Settings
from gws_sim import ODESystemHelper

settings = Settings.get_instance()


class Lorentz(ODESystemHelper):
    def state_names(self):
        """ state_names """
        return "u", "v", "w"

    def initial_state(self, args=None):
        """ initial_state """
        return (0, 1, 1.05)

    def parameters(self, t, args=None):
        """ parameters """
        return 10, 2.667, 28

    def derivative(self, t, x, args=None):
        """ derivative """
        u, v, w = x
        sigma, rho, beta = self.parameters(t, args)

        dudt = -sigma*(u - v)
        dvdt = rho*u - v - u*w
        dwdt = -beta*w + u*v
        return [dudt, dvdt, dwdt]


class TestLinearODESim(BaseTestCase):

    def test_nonlin(self):
        sys = Lorentz()
        sol = sys.simulate(t_start=0, t_end=100, t_step=0.05)

        self.assertEqual(sol.success, True)
        print(sol.y)
