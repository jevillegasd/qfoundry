"""Utility functions for energies, couplings, and superconducting parameters.

Some formulas reference:
- Jeffrey et al., Phys. Rev. Lett. 112, 190504 (2014)
- Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)
"""

from scipy.constants import elementary_charge as e_0
from scipy.constants import h, hbar
from scipy.constants import Boltzmann as k_B
from numpy import sqrt, pi, tanh, ix_
from numpy.linalg import inv

from qfoundry.resonator import circuit

# Helper function for capacitance operations
def parallel(Za, Zb):
    return 1 / (1/Za + 1/Zb)

def series(Za,Zb):
    return Za + Zb

def parallel_capacitance(Ca, Cb):
    return series(Ca, Cb)

def series_capacitance(Ca, Cb):
    return parallel(Ca, Cb)

def Schur_complement(C, indices_A, indices_B):
    """
    Calculate the Schur complement of a capacitance matrix C.
    C: Full capacitance matrix
    indices_A: Indices of the subsystem A to keep
    indices_B: Indices of the subsystem B to eliminate
    Returns the effective capacitance matrix for subsystem A.
    """

    C_AA = C[ix_(indices_A, indices_A)]
    C_BB = C[ix_(indices_B, indices_B)]
    C_AB = C[ix_(indices_A, indices_B)]
    C_BA = C[ix_(indices_B, indices_A)]

    C_eff = C_AA - C_AB @ inv(C_BB) @ C_BA
    return C_eff


def delta_cap(C12, C13, C23, C3):
    """
    Calculate the effective coupling capacitance between nodes 1 and 2
    using the delta-to-wye transformation.
    C12: Capacitance between nodes 1 and 2
    C13: Capacitance between nodes 1 and 3
    C23: Capacitance between nodes 2 and 3
    C3: Capacitance of node 3 to ground
    """

    return C12 + (C13*C23)/C3 # Neglects ground capacitance of nodes 1 and 2

def Cs_to_E(C):
    return e_0**2 / (2 * C) / h


def E_to_C(E):
    """
    Convert energy to capacitance.
    E = e^2/(2*C) => C = e^2/(2*E)
    """
    return e_0**2 / (2 * E) / h


def Cq_to_E(Cq, C1, C2):
    """
    Using Vi = (2e/Ci)ni and the Hamiltonian H_int = Hint = CgV1V2
    So that in the perturbative regime, H_int = g_int*n1*n2
    H_int = 4*e_0**2*Cq/(C1*C2)/h
    https://arxiv.org/pdf/1904.06560 (Krantz 2021) eq. 27
    """
    return 4 * e_0**2 * Cq / (C1 * C2 - Cq**2) / h


def cap_coupling(Ck, C1, C2, w1, w2):
    """
    The standard formula for capacitive coupling between harmonic modes [1].
    The entries w1 and w2 in radian frequencies.
    [1] E. Jeffrey, Phys. Rev. Lett. 112, 190504, https://arxiv.org/pdf/1401.0257
    """
    return 0.5 * sqrt(w1 * w2) * Ck / (sqrt(C1 * C2))


def C_to_g(Cg, C_sum, f0, Cr):
    # Wallraff et al. 2004
    return e_0 * Cg / (Cg + C_sum) * sqrt(2 * f0 / (h * Cr))


def g_hm(Cg, hm0: circuit, hm1: circuit):
    """
    Capacitive coupling between harmonic circuits.
    """
    # return Cg_to_E(Cg, hm0.C(), hm1.C())
    return cap_coupling(Cg, hm0.C(), hm1.C(), hm0.f0() * 2 * pi, hm1.f0() * 2 * pi)


class sc_metal:
    """
    Superconductive metal.
    Modelled only from its critical temperature.
    """

    def __init__(self, Tc=1.14, T=20e-3):
        self.Tc = Tc
        self.T = T

    def sc_gap(self):
        if self.T < 0.1:
            return 1.764 * k_B * self.Tc
        else:
            return 3.076 * k_B * sqrt(1 - self.T / self.Tc)

    def sc_gap_eV(self):
        return self.sc_gap() / e_0


def Ic_to_R(Ic, mat=sc_metal(1.14, T=20e-3)):
    """
    Convert Ic to R.
    R = pi*Delta/(2*e_0*Ic)*tanh(Delta/(2*k_B*T))
    where Delta is the superconducting gap.
    #https://www.pearsonhighered.com/assets/samplechapter/0/1/3/2/0132627426.pdf page 162
    """

    return (
        pi * mat.sc_gap() / (2 * e_0 * Ic) * tanh(mat.sc_gap() / (2 * k_B * mat.T))
    )


def R_to_Ic(R, mat=sc_metal(1.14, T=20e-3)):
    """
    Convert resistance to critical current.
    Ic = pi*Delta/(2*e_0*R_jx)*tanh(Delta/(2*k_B*T))
    where Delta is the superconducting gap.
    """
    return pi * mat.sc_gap() / (2 * e_0 * R) * tanh(mat.sc_gap() / (2 * k_B * mat.T))
