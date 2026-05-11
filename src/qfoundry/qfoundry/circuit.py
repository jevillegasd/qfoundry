"""Base RLC circuit model.

Provides the :class:`circuit` base class representing a general lumped-element
RLC network. This is the common ancestor of both resonator models
(:class:`~qfoundry.resonator.cpw_resonator`) and qubit models
(:class:`~qfoundry.qubits.transmon`).
"""

import numpy as np


class circuit:
    """
    A general RLC electrical circuit.

    Attributes
    ----------
        R:float=np.inf
            Circuit equivalent resistance in Ohms
        L:float = np.inf
            Circuit equivalent inductance in Henry
        C:float = 0.0
            Circuit equivalent capacitrance in Farads
        n:float = 0.0
            RCL resonance mode
        type: str = 'p'
            Circuit type. p is capacitor in parallel and s in series capacitor,

    Methods
    -------

    """

    def __init__(
        self,
        R: float = np.inf,
        L: float = np.inf,
        C: float = 0,
        n: float = 1,
        c_type: str = "p",
    ):  # Type is p for parallel RLC and s for series
        self._R_ = R
        self._L_ = L
        self._C_ = C
        self.n = n
        self.c_type = c_type

    def _Zs_(self, f):
        w = 2 * np.pi * f
        return self.R() + 1j * w * self.L() + 1 / (1j * w * self.C())

    def _Zp_(self, f):
        w = 2 * np.pi * f
        return 1 / (1 / self.R() + 1 / (1j * w * self.L() * self.n) + 1j * w * self.C() * self.n)

    def _f0_(self):
        return 1 / (2 * np.pi * np.sqrt(self.L() * self.C()))

    def Q(self):
        return self._R_ * np.sqrt(self.C() / self.L())

    def Z(self, f):
        """
        Frequency domain numeric transfer function (impedance)
        """
        if self.c_type == "s":
            return self._Zs_(f)
        else:
            return self._Zp_(f)

    def __add__(self, o):
        """Return a callable Z(f) = Z_self(f) + Z_other(f)."""
        return lambda f: self.Z(f) + o.Z(f)

    def __multiply__(self, o):
        """Return a callable Z(f) = Z_self(f) * Z_other(f)."""
        return lambda f: self.Z(f) * o.Z(f)

    def R(self):
        return self._R_

    def C(self):
        return self._C_

    def L(self):
        return self._L_
