# Gencovery software - All rights reserved
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


class Law:
    """ Mass action laws proxy """

    @staticmethod
    def mm(vmax: float, KM: float, X: float) -> float:
        """
        Michaelis-Menten mass action law

        The law is `f(x) = vmax * X / (KM + X)`

        :param k: Constant
        :type k: `float`
        :param vmax: Maximal reaction rate
        :type vmax: `float`
        :param X: State variable
        :type X: `float`
        :param KM: Half-saturation constant
        :type KM: `float`
        :returns: `float`
        """
        return vmax * X / (KM + X)

    @staticmethod
    def mm_inv(vmax: float, KM: float, X: float) -> float:
        """
        Inverse Michaelis-Menten mass action law

        The law is `f(x) = vmax * / (KM + X)`

        :param k: Constant
        :type k: `float`
        :param vmax: Maximal reaction rate
        :type vmax: `float`
        :param X: State variable
        :type X: `float`
        :param KM: Half-saturation constant
        :type KM: `float`
        :returns: `float`
        """
        return vmax / (KM + X)

    @staticmethod
    def hill(vmax: float, KH: float, n: float, X: float) -> float:
        """
        Hill mass action law

        The law is `f(x) = vmax * X^n / (KH^n + X^n)`

        :param n: Nonlinearity coefficient
        :type n: `float`
        :param k: Constant
        :type k: `float`
        :param vmax: Maximal reaction rate
        :type vmax: `float`
        :param X: State variable
        :type X: `float`
        :param KM: Half-saturation constant
        :type KM: `float`
        :returns: `float`
        """

        return vmax * X**n / (KH**n + X**n)

    @staticmethod
    def hill_inv(vmax: float, KH: float, n: float, X: float) -> float:
        """
        Inverse Hill mass action law

        The law is `f(x) = vmax / (KH^n + X^n)`

        :param n: Nonlinearity coefficient
        :type n: `float`
        :param k: Constant
        :type k: `float`
        :param vmax: Maximal reaction rate
        :type vmax: `float`
        :param X: State variable
        :type X: `float`
        :param KM: Half-saturation constant
        :type KM: `float`
        :returns: `float`
        """

        return vmax / (KM**n + X**n)
