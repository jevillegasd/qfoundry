"""Utility functions for energies, couplings, and superconducting parameters.

Some formulas reference:
- Jeffrey et al., Phys. Rev. Lett. 112, 190504 (2014)
- Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)
"""

from scipy.constants import elementary_charge as e_0
from scipy.constants import h, hbar
from numpy import sqrt, pi, tanh, ix_
from numpy.linalg import inv

from qfoundry.circuit import circuit
from qfoundry.materials import sc_metal  # re-exported for backward compatibility

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
    Convert charging energy E_C/h (Hz) to capacitance (F). Inverse of Cs_to_E.
    E_C/h = e^2/(2*C*h) => C = e^2/(2*(E_C/h)*h)
    """
    return e_0**2 / (2 * E) / h


def L_to_E(L):
    r"""Inductive energy (Hz) implied by an inductance L (H).

    Using the reduced flux quantum :math:`\varphi_0 = h/(2e) / 2\pi`, the inductive 
    energy is :math:`E_L = (\varphi_0)^2/L`. Using using the reduced flux quantum, 
    $\varphi_0 = \Phi_0 / 2\pi = \hbar / 2e$, is the only way to make the dimensionless 
    superconducting phase canonically conjugate to the Cooper pair number operator without
    introducing a factor of $2\pi$ in the commutation relation. The inductive energy is
    self-consistent with :math:`f_0 = \sqrt{8 E_C E_L}` (see
    :meth:`qfoundry.resonator.cpw_resonator.from_energies`) for a plain LC
    oscillator with :math:`f_0 = 1/(2\pi\sqrt{LC})`:

    .. math::

        E_L/h = \frac{\varphi_0^2}{L h} = \frac{\hbar^2}{4 e^2 L h}
    """
    phi_0 = hbar / (2 * e_0) # reduced flux quantum
    return phi_0**2 / L / h  # in Hz


def E_to_L(E):
    """
    Convert inductive energy (Hz) to inductance (H).
    E_L/h = phi_0^2/(L*h) => L = phi_0^2/(E_L*h)
    """
    phi_0 = hbar / (2 * e_0)  # reduced flux quantum
    return phi_0**2 / E / h  # in H


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


def Ic_to_R(Ic, mat=sc_metal(1.14, T=20e-3)):
    """
    Convert Ic to R.
    """
    from .josephson import JosephsonJunctionAnalyzer

    return JosephsonJunctionAnalyzer().Ic_to_R(Ic, mat.sc_gap(), T=mat.T)


def R_to_Ic(R, mat=sc_metal(1.14, T=20e-3)):
    """
    Convert resistance to critical current.
    """
    from .josephson import JosephsonJunctionAnalyzer
    return JosephsonJunctionAnalyzer().R_to_Ic(R, mat.sc_gap(), T=mat.T)


def Ej_to_Ic(Ej):
    """Critical current (A) implied by Josephson energy Ej (Hz). Ic = Ej*4*pi*e_0."""
    return Ej * 4.0 * pi * e_0


def Ic_to_Ej(Ic):
    """Josephson energy (Hz) implied by critical current Ic (A). Inverse of Ej_to_Ic."""
    return Ic / (4.0 * pi * e_0)


def Ck_to_kappa_ext(f0, Ck, C, Z_L=50.0):
    """External coupling rate (Hz) from coupling capacitance Ck onto a resonator
    of effective capacitance C at frequency f0. Same relation as
    cpw_resonator.kappa_ext(), exposed on raw scalars for pipelines (e.g.
    FEM-cap-matrix-derived C) that don't have a full cpw geometry object.

    kappa_ext = (2*pi*f0*Ck)^2 * Z_L / C / (2*pi)
    """
    omega0 = 2 * pi * f0
    return (omega0 * Ck) ** 2 * Z_L / C / (2 * pi)


def kappa_ext_to_Ck(f0, kappa_ext, C, Z_L=50.0):
    """Coupling capacitance Ck required for a target kappa_ext (Hz). Inverse of
    Ck_to_kappa_ext()."""
    omega0 = 2 * pi * f0
    return sqrt(kappa_ext * 2 * pi * C / Z_L) / omega0
