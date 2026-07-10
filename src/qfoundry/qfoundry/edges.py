"""QPU edge (coupler) models for qubit-qubit interactions.

Each edge is *directional*: ``edge(q0, q1)`` designates q0 as the control
qubit and q1 as the target.  Although the classical physical parameters of a
symmetric coupler are unchanged by swapping q0 and q1, the derived quantities
nu (IX) and mu (ZX) are asymmetric because the detuning sign and the
anharmonicity that enters each numerator/denominator differ.

Coupling types
--------------
* :class:`capacitive_coupler`  – fixed capacitive coupling via C_12
* :class:`inductive_coupler`   – fixed mutual-inductance coupling via M
* :class:`bus_resonator_coupler` – resonator-mediated exchange coupling
* :class:`tunable_coupler`     – flux-tunable SQUID-based coupler
* :class:`hybrid_coupler`      – 3-D integrated coupler (C_12 + M)

Module-level helper functions (qubit-specific, not imported from utils)
------------------------------------------------------------------------
* :func:`_g_cap_qq`   – capacitive qubit–qubit coupling strength (Hz)
* :func:`_g_ind_qq`   – inductive qubit–qubit coupling strength (Hz)
* :func:`_g_cap_qr`   – capacitive qubit–resonator coupling strength (Hz)
* :func:`_I_zpf`      – thin wrapper; delegates to :meth:`~qubits.qubit.I_zpf`

References
----------
[Blais2021]      Blais, Grimsmo, Girvin, Wallraff,
                 Rev. Mod. Phys. 93, 025005 (2021).
[Krantz2019]     Krantz et al., Appl. Phys. Rev. 6, 021318 (2019).
[Majer2007]      Majer et al., Nature 449, 443 (2007).
[Koch2007]       Koch et al., Phys. Rev. A 76, 042319 (2007).
[Magesan2020]    Magesan & Gambetta, Phys. Rev. A 101, 052308 (2020).
[Chen2014]       Chen et al., Phys. Rev. Lett. 113, 220502 (2014).
[Yan2018]        Yan et al., Phys. Rev. Applied 10, 054062 (2018).
[Rosenberg2017]  Rosenberg et al., npj Quantum Inf. 3, 42 (2017).
[Ku2020]         Ku et al., PRX Quantum 2, 040305 (2020).
[Jeffrey2014]    Jeffrey et al., Phys. Rev. Lett. 112, 190504 (2014).
[Wallraff2004]   Wallraff et al., Nature 431, 162 (2004).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.constants import h, hbar, e, pi
from scipy.optimize import brentq

from qfoundry.qubits import transmon
from qfoundry.resonator import cpw_resonator
import scqubits as scq

# ---------------------------------------------------------------------------
# Module-level coupling helper functions
# ---------------------------------------------------------------------------

def _I_zpf(qubit) -> float:
    """Zero-point current fluctuation of a qubit (A).

    Delegates to :meth:`qubit.I_zpf` defined in the :mod:`qubits` module.
    Kept here as a convenience wrapper for backward compatibility.

    Parameters
    ----------
    qubit :
        Any object exposing an ``I_zpf()`` method (e.g. :class:`~qubits.transmon`).

    Returns
    -------
    float
        :math:`I_{\\mathrm{zpf}}` in Amperes.

    References
    ----------
    [Blais2021] Eq. (2.22); [Krantz2019] Eq. (18).
    """
    return qubit.I_zpf()


def _g_cap_qq(C_12: float, q0: transmon, q1: transmon) -> float:
    r"""Capacitive coupling strength between two transmon-like qubits (Hz).

    Derived from the Jaynes–Cummings interaction between two charge-basis
    modes coupled through a capacitance :math:`C_{12}`:

    .. math::

        g = \frac{C_{12}}{2\sqrt{C_{\Sigma,0}\, C_{\Sigma,1}}}
            \sqrt{\omega_{01}^{(0)}\, \omega_{01}^{(1)}}

    This is the harmonic-mode approximation.  For the full transmon matrix
    element the charge matrix element :math:`n_{01}` can replace the
    harmonic ZPF factor, but the harmonic form is accurate in the transmon
    regime (:math:`E_J / E_C \gg 1`).

    Parameters
    ----------
    C_12 : float
        Coupling capacitance between the two qubits in Farads.
    q0, q1 : transmon
        Qubit objects exposing ``C()`` (F) and ``omega01()`` (rad s⁻¹).

    Returns
    -------
    float
        Coupling strength :math:`g / (2\pi)` in Hz.

    References
    ----------
    [Blais2021] Eq. (4.9); [Krantz2019] Eq. (27).
    """
    C0    = q0.C()         # F
    C1    = q1.C()         # F
    f0    = q0.f01()   # rad/s
    f1    = q1.f01()   # rad/s
    g_Hz = 0.5 * np.sqrt(f0 * f1) * C_12 / np.sqrt(C0 * C1)
    return g_Hz


def _g_ind_qq(M: float, q0: transmon, q1: transmon) -> float:
    r"""Inductive (mutual-inductance) coupling strength between two qubits (Hz).

    .. math::

        g = \frac{M \, I_{\mathrm{zpf},0} \, I_{\mathrm{zpf},1}}{h}

    where :math:`I_{\mathrm{zpf},i} = \sqrt{\hbar \omega_{01}^{(i)} / (2 L_{J,i})}`.

    Parameters
    ----------
    M : float
        Mutual inductance in Henries.
    q0, q1 :
        Qubit objects exposing ``omega01()`` (rad s⁻¹) and ``L()`` (H).

    Returns
    -------
    float
        Coupling strength in Hz.

    References
    ----------
    [Blais2021] §2.3; [Krantz2019] §2.2.
    """
    I0 = _I_zpf(q0)
    I1 = _I_zpf(q1)
    return M * I0 * I1 / h


def _g_cap_qr(C_qr: float, qubit: transmon, resonator: cpw_resonator) -> float:
    r"""Capacitive qubit–resonator coupling strength (Hz).

    .. math::

        g_{qr} = \frac{C_{qr}}{2\sqrt{C_{\Sigma,q}\, C_r}}
                 \sqrt{\omega_q \, \omega_r}

    Parameters
    ----------
    C_qr : float
        Coupling capacitance between qubit and resonator in Farads.
    qubit :
        Qubit object exposing ``C()`` and ``omega01()``.
    resonator :
        Resonator object exposing ``C()`` and ``f0()`` (Hz).

    Returns
    -------
    float
        Coupling strength in Hz.

    References
    ----------
    [Blais2021] Eq. (4.9); [Wallraff2004] Nature 431, 162.
    """
    C_q   = qubit.C()
    C_r   = resonator.C()
    w_q   = qubit.omega01()
    w_r   = resonator.f0() * 2.0 * pi
    g_rads = 0.5 * np.sqrt(w_q * w_r) * C_qr / np.sqrt(C_q * C_r)
    return g_rads / (2.0 * pi)


def _C_cap_qr(g_Hz: float, qubit: transmon, resonator: cpw_resonator) -> float:
    r"""Coupling capacitance implied by a target qubit–resonator coupling (F).

    Inverse of :func:`_g_cap_qr`:

    .. math::

        C_{qr} = 2\,g_{qr}\,\frac{\sqrt{C_{\Sigma,q}\, C_r}}{\sqrt{f_q\, f_r}}

    Parameters
    ----------
    g_Hz : float
        Target qubit–resonator coupling strength (Hz).
    qubit :
        Qubit object exposing ``C()`` and ``f01()``.
    resonator :
        Resonator object exposing ``C()`` and ``f0()`` (Hz).

    Returns
    -------
    float
        Coupling capacitance in Farads.

    References
    ----------
    [Blais2021] Eq. (4.9); [Wallraff2004] Nature 431, 162.
    """
    C_q = qubit.C()
    C_r = resonator.C()
    f_q = qubit.f01()
    f_r = resonator.f0()
    return 2.0 * g_Hz * np.sqrt(C_q * C_r) / np.sqrt(f_q * f_r)


def label_bare_states(evecs: np.ndarray, dims: list) -> list:
    r"""Identify the dominant bare product state for each dressed eigenvector.

    For a composite :class:`scqubits.HilbertSpace` built as a tensor product
    of subsystems (each already expressed in its own diagonalized/truncated
    eigenbasis — e.g. ``[q0, resonator, q1]`` as built by
    :meth:`bus_resonator_coupler.hilbert_space`), the bare product state
    :math:`|n_0, n_1, \ldots\rangle` is exactly the standard basis vector at
    the flattened (row-major/``kron``-order) index. So the bare state with
    maximum overlap for a given dressed eigenvector is simply the basis index
    of its largest-magnitude component — no need to construct and project
    against candidate bare vectors one at a time.

    Parameters
    ----------
    evecs : ndarray
        Dressed eigenvectors as columns, shape ``(dim, n_states)`` — the
        ``evecs`` returned by :meth:`bus_resonator_coupler.dressed_eigensys`.
    dims : list of int
        Truncated dimension of each subsystem, in the same order used to
        build the composite Hilbert space (e.g.
        ``[s.truncated_dim for s in hs.subsystem_list]``).

    Returns
    -------
    list of tuple of int
        One bare-state tuple per column of ``evecs``, in the same subsystem
        order as ``dims``.
    """
    labels = []
    for col in range(evecs.shape[1]):
        probs = np.abs(evecs[:, col]) ** 2
        idx = int(np.argmax(probs))
        tup = []
        for d in reversed(dims):
            idx, n = divmod(idx, d)
            tup.append(n)
        labels.append(tuple(reversed(tup)))
    return labels


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class edge(ABC):
    """Abstract base class for all QPU coupler edges.

    An edge is *directional*: ``q0`` is the control qubit and ``q1`` is the
    target.  Swapping them changes the sign of the detuning and therefore the
    cross-resonance interaction coefficients :meth:`nu` and :meth:`mu`,
    making ``edge(q0, q1)`` physically distinct from ``edge(q1, q0)``.

    Parameters
    ----------
    q0 :
        Control qubit (driven qubit in cross-resonance notation).
    q1 :
        Target qubit.

    Attributes
    ----------
    q0, q1 : qubit objects

    Notes
    -----
    All frequencies are returned in Hz (matching the qfoundry convention).
    Angular frequencies used internally are expressed in rad s⁻¹.
    """

    def __init__(self, q0, q1):
        self.q0 = q0
        self.q1 = q1

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def g(self) -> float:
        """Transverse (exchange) coupling strength in Hz.

        Subclasses must implement this method according to their coupling
        mechanism.
        """

    @abstractmethod
    def hilbert_space(self) -> scq.HilbertSpace:
        """Build and return a :class:`scqubits.HilbertSpace` for this edge.

        For two-body (direct) couplers the space contains only q0 and q1.
        For three-body (mediated) couplers the coupler element is included.
        """

    # ------------------------------------------------------------------
    # Derived two-qubit interaction coefficients
    # ------------------------------------------------------------------

    def zeta(self) -> float:
        r"""ZZ coupling coefficient :math:`\zeta` (Hz).

        Second-order perturbative expression for the static ZZ interaction
        between the two qubits:

        .. math::

            \zeta = \frac{2g^2 \alpha_0}{\Delta(\Delta + \alpha_0)}
                  + \frac{2g^2 \alpha_1}{-\Delta(-\Delta + \alpha_1)}

        where :math:`\Delta = (\omega_0 - \omega_1)/(2\pi)` (Hz) is the bare
        qubit detuning, :math:`\alpha_i` is the anharmonicity of qubit *i*
        (Hz), and :math:`g` is the transverse coupling (Hz).

        Note that :math:`\zeta` is symmetric under q0 ↔ q1 by construction.

        References
        ----------
        [Blais2021] Eq. (4.28); [Ku2020] supplementary material.
        """
        g_hz  = self.g()
        a0    = self.q0.alpha()   # Hz  (negative for transmon)
        a1    = self.q1.alpha()   # Hz
        Delta = (self.q0.omega01() - self.q1.omega01()) / (2.0 * pi)  # Hz

        if Delta == 0.0:
            raise ValueError("zeta() is undefined for degenerate qubits (Δ = 0).")

        term0 = 2.0 * g_hz**2 * a0 / (Delta * (Delta + a0))
        term1 = 2.0 * g_hz**2 * a1 / (-Delta * (-Delta + a1))
        return term0 + term1

    def nu(self) -> float:
        r"""Bare IX coupling coefficient :math:`\nu` (Hz per rad s⁻¹ of drive).

        In the cross-resonance (CR) interaction frame where q0 is driven at
        the frequency of q1, the leading-order IX term scales as:

        .. math::

            \nu = \frac{g \, \alpha_1}{\Delta_{01}(\Delta_{01} + \alpha_1)}

        where :math:`\Delta_{01} = f_0 - f_1` (Hz) and :math:`\alpha_1` is
        the anharmonicity of the *target* qubit.  The full IX rate under a
        drive of amplitude :math:`\Omega` (rad s⁻¹) is :math:`\nu \cdot \Omega`.

        The sign (and magnitude) depends on the control/target assignment:
        ``edge(q0, q1).nu()`` ≠ ``edge(q1, q0).nu()`` in general.

        References
        ----------
        [Magesan2020] Eq. (C8).
        """
        g_hz   = self.g()
        a1     = self.q1.alpha()
        Delta  = (self.q0.omega01() - self.q1.omega01()) / (2.0 * pi)  # Hz

        if Delta == 0.0 or (Delta + a1) == 0.0:
            raise ValueError("nu() is singular for this qubit pair (Δ=0 or Δ+α₁=0).")

        return g_hz * a1 / (Delta * (Delta + a1))

    def mu(self) -> float:
        r"""Bare ZX coupling coefficient :math:`\mu` (Hz per rad s⁻¹ of drive).

        Second-order contribution to the ZX interaction in the CR frame:

        .. math::

            \mu = \frac{g^2}{\Delta_{01}}
                  \left(\frac{1}{\Delta_{01} + \alpha_0}
                       -\frac{1}{\Delta_{01} - \alpha_1}\right)

        where :math:`\alpha_0` is the anharmonicity of the *control* qubit.
        The full ZX rate under a drive :math:`\Omega` is :math:`\mu \cdot \Omega`.

        References
        ----------
        [Magesan2020] Eq. (C9).
        """
        g_hz  = self.g()
        a0    = self.q0.alpha()
        a1    = self.q1.alpha()
        Delta = (self.q0.omega01() - self.q1.omega01()) / (2.0 * pi)  # Hz

        if Delta == 0.0:
            raise ValueError("mu() is singular for degenerate qubits (Δ = 0).")

        return g_hz**2 / Delta * (1.0 / (Delta + a0) - 1.0 / (Delta - a1))

    def nu_driven(self, Omega: float) -> float:
        r"""IX interaction rate under a CR drive of amplitude :math:`\Omega`.

        .. math::

            \Omega_{IX} = \nu \cdot \Omega

        Parameters
        ----------
        Omega : float
            Drive amplitude in rad s⁻¹.

        Returns
        -------
        float
            IX rate in Hz.

        References
        ----------
        [Magesan2020] Eq. (C8).
        """
        return self.nu() * Omega

    def mu_driven(self, Omega: float) -> float:
        r"""ZX interaction rate under a CR drive of amplitude :math:`\Omega`.

        .. math::

            \Omega_{ZX} = \mu \cdot \Omega

        Parameters
        ----------
        Omega : float
            Drive amplitude in rad s⁻¹.

        Returns
        -------
        float
            ZX rate in Hz.

        References
        ----------
        [Magesan2020] Eq. (C9).
        """
        return self.mu() * Omega

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_qmodels(self):
        """Assert that both qubits have an instantiated scqubits qmodel."""
        if self.q0.qmodel is None:
            raise ValueError("q0.qmodel is not instantiated.  Pass inst_model=True.")
        if self.q1.qmodel is None:
            raise ValueError("q1.qmodel is not instantiated.  Pass inst_model=True.")

    def __str__(self) -> str:
        try:
            g_val    = self.g()
            zeta_val = self.zeta()
            nu_val   = self.nu()
            mu_val   = self.mu()
            return (
                f"{self.__class__.__name__}({self.q0.__class__.__name__} → "
                f"{self.q1.__class__.__name__})\n"
                f"  g    = {g_val * 1e-6:+.3f} MHz\n"
                f"  zeta = {zeta_val * 1e-6:+.3f} MHz\n"
                f"  nu   = {nu_val:.3e}  (IX/rad·s⁻¹)\n"
                f"  mu   = {mu_val:.3e}  (ZX/rad·s⁻¹)"
            )
        except Exception as exc:  # noqa: BLE001
            return f"{self.__class__.__name__}: {exc}"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Concrete coupler classes
# ---------------------------------------------------------------------------

class capacitive_coupler(edge):
    """Fixed capacitive coupler between two qubits.

    The two qubits are coupled through a shunt capacitance :math:`C_{12}`.
    This is the dominant coupling mechanism in most planar transmon circuits,
    realised by interdigitated finger capacitors or gap coupling.

    Parameters
    ----------
    q0 : transmon-like
        Control qubit.
    q1 : transmon-like
        Target qubit.
    C_12 : float
        Coupling capacitance in Farads.

    Notes
    -----
    The coupling Hamiltonian is :math:`H_{int} = g (a_0 + a_0^\dagger)(a_1 + a_1^\dagger)`,
    which in the rotating-wave approximation reduces to the Jaynes–Cummings
    exchange interaction.  In the scqubits HilbertSpace it is implemented
    via the charge operator :math:`\\hat{n}`.

    References
    ----------
    [Blais2021] §4.2; [Krantz2019] Eq. (27).
    """

    def __init__(self, q0, q1, C_12: float):
        super().__init__(q0, q1)
        self.C_12 = C_12

    def g(self) -> float:
        r"""Capacitive qubit–qubit coupling strength (Hz).

        .. math::

            g = \frac{C_{12}}{2\sqrt{C_{\Sigma,0}\,C_{\Sigma,1}}}
                \sqrt{\omega_{01}^{(0)}\,\omega_{01}^{(1)}}

        Returns
        -------
        float
            Coupling strength in Hz.

        References
        ----------
        [Blais2021] Eq. (4.9); [Krantz2019] Eq. (27).
        """
        return _g_cap_qq(self.C_12, self.q0, self.q1)

    def hilbert_space(self) -> scq.HilbertSpace:
        """Build 2-body HilbertSpace with capacitive n-n interaction.

        The interaction is

        .. math::

            H_{int} = g \\cdot \\hat{n}_0 \\otimes \\hat{n}_1

        expressed in units of h (Hz), consistent with the scqubits EJ/EC
        convention used in qfoundry.

        Returns
        -------
        scqubits.HilbertSpace
        """
        self._check_qmodels()
        hs = scq.HilbertSpace([self.q0.qmodel, self.q1.qmodel])
        g = self.g()
        hs.add_interaction(
            g_strength=g * 1e-9,   # GHz to match scqubits energy units
            op1=(self.q0.qmodel.n_operator, self.q0.qmodel),
            op2=(self.q1.qmodel.n_operator, self.q1.qmodel),
            add_hc=True,
        )
        return hs


class inductive_coupler(edge):
    """Fixed mutual-inductance (inductive) coupler between two qubits.

    The coupling is mediated by mutual inductance :math:`M` between the
    Josephson inductances of the two qubits.  This is commonly implemented
    as shared kinetic inductance or a geometric loop.

    Parameters
    ----------
    q0 : transmon-like
        Control qubit.
    q1 : transmon-like
        Target qubit.
    M : float
        Mutual inductance in Henries.

    Notes
    -----
    The coupling Hamiltonian is :math:`H_{int} = g \\, \\hat{\\varphi}_0 \\hat{\\varphi}_1`
    where :math:`\\hat{\\varphi}` is the dimensionless flux (phase) operator.
    In scqubits this is the ``phi_operator``.

    References
    ----------
    [Blais2021] §2.3; [Krantz2019] §2.2.
    """

    def __init__(self, q0, q1, M: float):
        super().__init__(q0, q1)
        self.M = M

    def g(self) -> float:
        r"""Inductive qubit–qubit coupling strength (Hz).

        .. math::

            g = \frac{M\,I_{\mathrm{zpf},0}\,I_{\mathrm{zpf},1}}{h}

        where :math:`I_{\mathrm{zpf},i} = \sqrt{\hbar\,\omega_{01}^{(i)} / (2 L_{J,i})}`.

        Returns
        -------
        float
            Coupling strength in Hz.

        References
        ----------
        [Blais2021] §2.3; [Krantz2019] §2.2.
        """
        return _g_ind_qq(self.M, self.q0, self.q1)

    def hilbert_space(self) -> scq.HilbertSpace:
        """Build 2-body HilbertSpace with inductive phi-phi interaction.

        The interaction is

        .. math::

            H_{int} = g \\cdot \\hat{\\varphi}_0 \\otimes \\hat{\\varphi}_1

        Returns
        -------
        scqubits.HilbertSpace
        """
        self._check_qmodels()
        hs = scq.HilbertSpace([self.q0.qmodel, self.q1.qmodel])
        g = self.g()
        hs.add_interaction(
            g_strength=g * 1e-9,
            op1=(self.q0.qmodel.phi_operator, self.q0.qmodel),
            op2=(self.q1.qmodel.phi_operator, self.q1.qmodel),
            add_hc=True,
        )
        return hs


class bus_resonator_coupler(edge):
    """Bus-resonator mediated coupling between two qubits.

    A coplanar waveguide (CPW) bus resonator mediates an effective coupling
    between the two qubits via virtual photon exchange.  Each qubit couples
    to the resonator through its own coupling capacitance.

    Parameters
    ----------
    q0 : transmon-like
        Control qubit.
    q1 : transmon-like
        Target qubit.
    resonator : cpw_resonator
        The bus resonator instance.
    C_0r : float
        Coupling capacitance between q0 and the resonator (F).
    C_1r : float
        Coupling capacitance between q1 and the resonator (F).

    Notes
    -----
    The effective exchange coupling is derived in the dispersive regime
    (:math:`|g_{ir}| \\ll |\\Delta_{ir}|`) by adiabatically eliminating the
    resonator mode.

    References
    ----------
    [Majer2007] Nature 449, 443 (2007);
    [Blais2021] Rev. Mod. Phys. 93, 025005, Eq. (136).
    """

    def __init__(self, q0, q1, resonator, C_0r: float, C_1r: float):
        super().__init__(q0, q1)
        self.resonator = resonator
        self.C_0r = C_0r
        self.C_1r = C_1r
        self._g_0r = None
        self._g_1r = None

    @classmethod
    def from_energies(
        cls,
        q0,
        q1,
        E_c: float,
        E_l: float,
        g0: float,
        g1: float,
        **resonator_kwargs,
    ) -> "bus_resonator_coupler":
        r"""Build a bus_resonator_coupler from resonator energies and g's.

        Alternate constructor for use when the bus resonator is specified by
        its characteristic energies rather than a pre-built
        :class:`~qfoundry.resonator.cpw_resonator` instance and coupling
        capacitances.  A resonator is instantiated via
        :meth:`cpw_resonator.from_energies`, whose frequency is

        .. math::

            f_r = \sqrt{8\,E_C\,E_L}

        and the qubit–resonator coupling strengths :math:`g_{0r}`,
        :math:`g_{1r}` are taken directly (Hz), bypassing the
        capacitance-based derivation used by the default constructor.

        Parameters
        ----------
        q0, q1 : transmon-like
            Control / target qubits.
        E_c : float
            Resonator charging energy :math:`E_C/h` (Hz).
        E_l : float
            Resonator inductive energy :math:`E_L/h` (Hz).
        g0, g1 : float
            Qubit–resonator coupling strengths :math:`g_{0r}`,
            :math:`g_{1r}` (Hz).
        **resonator_kwargs
            Additional keyword arguments forwarded to
            :meth:`cpw_resonator.from_energies` (e.g. ``wg``, ``n``,
            ``truncated_dim``). ``length_f`` defaults to 2 (half-wave) —
            a qubit-qubit coupling bus has both ends open (capacitively
            coupled to a qubit), unlike a quarter-wave readout resonator
            which requires a physically shorted end.

        Returns
        -------
        bus_resonator_coupler
        """
        resonator_kwargs.setdefault("length_f", 2)
        resonator = cpw_resonator.from_energies(E_c, E_l, **resonator_kwargs)
        obj = cls.__new__(cls)
        edge.__init__(obj, q0, q1)
        obj.resonator = resonator
        obj.C_0r = _C_cap_qr(g0, q0, resonator)
        obj.C_1r = _C_cap_qr(g1, q1, resonator)
        obj._g_0r = g0
        obj._g_1r = g1
        return obj

    @classmethod
    def from_frequency(
        cls,
        q0,
        q1,
        frequency: float,
        g0: float,
        g1: float,
        **resonator_kwargs,
    ) -> "bus_resonator_coupler":
        r"""Build a bus_resonator_coupler from a target frequency and g's.

        Alternate constructor for when the bus resonator is specified purely
        by its target resonance frequency rather than by :math:`E_C`/:math:`E_L`
        directly — useful when the physical CPW geometry (and hence the
        implied :math:`E_C`, :math:`E_L`) is not yet known analytically. A
        resonator is instantiated via :meth:`cpw_resonator.from_frequency`,
        which solves for the physical length (and hence :math:`E_C`,
        :math:`E_L`) from the default/given CPW geometry alone. The implied
        energies are recoverable from the resulting resonator via
        :func:`~qfoundry.utils.Cs_to_E` (``resonator.C()``) and
        :func:`~qfoundry.utils.L_to_E` (``resonator.L()``).

        Parameters
        ----------
        q0, q1 : transmon-like
            Control / target qubits.
        frequency : float
            Target bus resonator frequency (Hz).
        g0, g1 : float
            Qubit–resonator coupling strengths :math:`g_{0r}`,
            :math:`g_{1r}` (Hz).
        **resonator_kwargs
            Additional keyword arguments forwarded to
            :meth:`cpw_resonator.from_frequency` (e.g. ``wg``, ``n``,
            ``truncated_dim``). ``length_f`` defaults to 2 (half-wave; see
            :meth:`from_energies`).

        Returns
        -------
        bus_resonator_coupler
        """
        resonator_kwargs.setdefault("length_f", 2)
        resonator = cpw_resonator.from_frequency(frequency, **resonator_kwargs)
        obj = cls.__new__(cls)
        edge.__init__(obj, q0, q1)
        obj.resonator = resonator
        obj.C_0r = _C_cap_qr(g0, q0, resonator)
        obj.C_1r = _C_cap_qr(g1, q1, resonator)
        obj._g_0r = g0
        obj._g_1r = g1
        return obj

    @classmethod
    def from_maxwell(
        cls,
        q0,
        q1,
        nodes: list,
        C_matrix_F,
        q0_node: str,
        resonator_node: str,
        q1_node: str,
        E_l: float,
        **resonator_kwargs,
    ) -> "bus_resonator_coupler":
        r"""Build a bus_resonator_coupler from a Maxwell capacitance matrix.

        Alternate constructor for use when the qubit-resonator coupling
        capacitances and the resonator's own charging energy are not known
        analytically but have been extracted from a FEM (e.g. Ansys HFSS/Q3D)
        simulation as a full mutual-capacitance matrix over all islands in the
        coupling region.

        The matrix is reduced via Schur complement (adiabatic elimination of
        every node other than the two qubit islands and the resonator body),
        yielding a 3x3 effective capacitance matrix.  The off-diagonal terms
        give :math:`C_{0r}`, :math:`C_{1r}`; the resonator's own diagonal term
        gives its self-capacitance, from which :math:`E_C` follows via
        :func:`~qfoundry.utils.Cs_to_E`.

        :math:`E_L` is not derivable from a capacitance matrix alone and must
        be supplied directly (e.g. from geometry or a separate FEM inductance
        extraction).

        Parameters
        ----------
        q0, q1 : transmon-like
            Control / target qubits.
        nodes : list of str
            Ordered node names matching the rows/columns of ``C_matrix_F``.
        C_matrix_F : array_like
            Full NxN mutual-capacitance matrix in Farads, Maxwell convention
            (diagonal = self-cap to ground + sum of mutual caps; off-diagonal
            negative).
        q0_node, resonator_node, q1_node : str
            Names (within ``nodes``) of the q0 island, the resonator body, and
            the q1 island respectively.
        E_l : float
            Resonator inductive energy :math:`E_L/h` (Hz).
        **resonator_kwargs
            Additional keyword arguments forwarded to
            :meth:`cpw_resonator.from_energies`. ``length_f`` defaults to 2
            (half-wave) — a qubit-qubit coupling bus has both ends open
            (capacitively coupled to a qubit), unlike a quarter-wave readout
            resonator which requires a physically shorted end.

        Returns
        -------
        bus_resonator_coupler
        """
        from qfoundry.utils import Schur_complement, Cs_to_E

        i0, ir, i1 = nodes.index(q0_node), nodes.index(resonator_node), nodes.index(q1_node)
        M = np.asarray(C_matrix_F, dtype=float)
        env = [i for i in range(len(nodes)) if i not in (i0, ir, i1)]
        S = Schur_complement(M, [i0, ir, i1], env) if env else M[np.ix_([i0, ir, i1], [i0, ir, i1])]

        C_0r = abs(float(S[0, 1]))
        C_1r = abs(float(S[2, 1]))
        E_c = Cs_to_E(float(S[1, 1]))

        resonator_kwargs.setdefault("length_f", 2)
        resonator = cpw_resonator.from_energies(E_c, E_l, **resonator_kwargs)
        return cls(q0, q1, resonator, C_0r, C_1r)

    def _g_qr(self, qubit, C_qr: float) -> float:
        """Individual qubit–resonator coupling (Hz).

        References
        ----------
        [Blais2021] Rev. Mod. Phys. 93, 025005); [Wallraff2004] Nature 431, 162.
        """
        return _g_cap_qr(C_qr, qubit, self.resonator)

    def _g0r(self) -> float:
        """Qubit0–resonator coupling (Hz): direct value if set, else derived from C_0r."""
        return self._g_0r if self._g_0r is not None else self._g_qr(self.q0, self.C_0r)

    def _g1r(self) -> float:
        """Qubit1–resonator coupling (Hz): direct value if set, else derived from C_1r."""
        return self._g_1r if self._g_1r is not None else self._g_qr(self.q1, self.C_1r)

    def g(self) -> float:
        r"""Effective bus-mediated qubit–qubit coupling (Hz).

        In the dispersive limit the resonator is adiabatically eliminated,
        yielding:

        .. math::

            g_{\mathrm{eff}} = \frac{g_{0r}\,g_{1r}}{2}
                \left(\frac{1}{\Delta_{0r}} + \frac{1}{\Delta_{1r}}\right)

        where :math:`\Delta_{ir} = f_i - f_r` (Hz) is the qubit–resonator
        detuning.

        Returns
        -------
        float
            Effective coupling strength in Hz.

        References
        ----------
        [Majer2007] Nature 449, 443; [Blais2021] Eq. (6.2).
        """
        g_0r = self._g0r()
        g_1r = self._g1r()
        f_r  = self.resonator.f0()
        Delta_0r = self.q0.f01() - f_r   # Hz
        Delta_1r = self.q1.f01() - f_r   # Hz

        if Delta_0r == 0.0 or Delta_1r == 0.0:
            raise ValueError("g() is singular when a qubit is resonant with the bus.")

        return 0.5 * g_0r * g_1r * (1.0 / Delta_0r + 1.0 / Delta_1r)

    def J(self) -> float:
        return self.g()  # alias for consistency with Blais2021 notation

    def hilbert_space(self) -> scq.HilbertSpace:
        """Build 3-body HilbertSpace [q0, resonator, q1].

        Two capacitive n-operator interaction terms are added:
        q0–resonator and resonator–q1.

        Returns
        -------
        scqubits.HilbertSpace
        """
        self._check_qmodels()
        if self.resonator.qmodel is None:
            raise ValueError("resonator.qmodel is not instantiated.")

        hs = scq.HilbertSpace(
            [self.q0.qmodel, self.resonator.qmodel, self.q1.qmodel]
        )
        g_0r = self._g0r()
        g_1r = self._g1r()

        hs.add_interaction(
            g_strength=g_0r * 1e-9,
            op1=(self.q0.qmodel.n_operator, self.q0.qmodel),
            op2=(self.resonator.qmodel.creation_operator, self.resonator.qmodel),
            add_hc=True,
        )
        hs.add_interaction(
            g_strength=g_1r * 1e-9,
            op1=(self.resonator.qmodel.creation_operator, self.resonator.qmodel),
            op2=(self.q1.qmodel.n_operator, self.q1.qmodel),
            add_hc=True,
        )
        return hs

    def dressed_eigensys(self, flux=None, tunable_qubit=None, evals_count=10):
        r"""Diagonalize the full 3-body qubit–resonator–qubit Hamiltonian.

        Near a qubit–qubit resonance (e.g. the |11⟩–|02⟩ crossing used for a
        CZ gate) the perturbative :meth:`g`/:meth:`zeta`/:meth:`nu`/:meth:`mu`
        formulas are singular or inaccurate — their denominators are exactly
        the detuning being driven to zero. This method instead builds and
        diagonalizes the true composite Hamiltonian via :meth:`hilbert_space`,
        giving the dressed eigenspectrum directly.

        Parameters
        ----------
        flux : float, optional
            Reduced flux bias to apply to ``tunable_qubit`` before
            diagonalizing (e.g. the CZ operating point). Requires
            ``tunable_qubit`` to be given.
        tunable_qubit :
            Whichever of ``q0``/``q1`` is a flux-tunable transmon; its
            ``qmodel.flux`` is set to ``flux`` before building the Hilbert
            space. Ignored if ``flux`` is ``None``.
        evals_count : int, optional
            Number of dressed eigenstates to compute. Default 10.

        Returns
        -------
        evals : ndarray
            Dressed eigenvalues (GHz), ascending.
        evecs : ndarray
            Dressed eigenvectors as columns, shape ``(dim, evals_count)``, in
            the tensor-product basis ``[q0, resonator, q1]`` (each factor in
            its own truncated eigenbasis — see :meth:`avoided_crossing_gap`).
        hs : scq.HilbertSpace
            The diagonalized Hilbert space (e.g. for ``hs.subsystem_list``).
        """
        if flux is not None:
            if tunable_qubit is None:
                raise ValueError("tunable_qubit must be given when flux is specified.")
            tunable_qubit.qmodel.flux = flux

        hs = self.hilbert_space()
        hs.generate_lookup()
        evals, evecs_raw = hs.eigensys(evals_count=evals_count)
        if hasattr(evecs_raw[0], "full"):
            evecs = np.column_stack([
                np.array(v.full(), dtype=complex).reshape(-1) for v in evecs_raw
            ])
        else:
            evecs = np.array(evecs_raw, dtype=complex)
            if evecs.shape[0] == evals_count and evecs.ndim == 2:
                evecs = evecs.T
        return evals, evecs, hs

    def avoided_crossing_gap(
        self, bare_a, bare_b, flux=None, tunable_qubit=None, evals_count=10,
    ) -> float:
        r"""Frequency gap between the dressed states nearest two bare product
        states (Hz).

        Identifies each bare product state ``|n_{q0}, n_{res}, n_{q1}\rangle``
        by maximum overlap with the dressed eigenvectors from
        :meth:`dressed_eigensys` (standard bare-to-dressed labeling at an
        avoided crossing — each factor's bare state is the corresponding
        one-hot vector in that subsystem's own truncated eigenbasis, since
        :class:`scqubits.HilbertSpace` already represents the composite space
        as a tensor product of each subsystem's diagonalized eigenbasis).

        At an exact avoided crossing the two overlaps are close to 0.5/0.5
        and the returned gap equals the full splitting — the effective
        coupling strength there is ``gap / 2``.

        Parameters
        ----------
        bare_a, bare_b : tuple of int
            Bare occupation tuples ``(n_q0, n_resonator, n_q1)`` for the two
            states of interest (e.g. ``(1, 0, 1)`` for |11⟩ and ``(2, 0, 0)``
            for |02⟩ of q0).
        flux, tunable_qubit, evals_count :
            Forwarded to :meth:`dressed_eigensys`.

        Returns
        -------
        float
            ``|E_dressed(bare_b) - E_dressed(bare_a)|`` in Hz.
        """
        evals, evecs, hs = self.dressed_eigensys(
            flux=flux, tunable_qubit=tunable_qubit, evals_count=evals_count,
        )
        dims = [s.truncated_dim for s in hs.subsystem_list]

        def _bare_vector(bare_tuple):
            vecs = []
            for n, d in zip(bare_tuple, dims):
                v = np.zeros(d, dtype=complex)
                v[n] = 1.0
                vecs.append(v)
            full = vecs[0]
            for v in vecs[1:]:
                full = np.kron(full, v)
            return full

        def _dressed_energy(bare_tuple):
            overlaps = np.abs(evecs.conj().T @ _bare_vector(bare_tuple)) ** 2
            return evals[int(np.argmax(overlaps))]

        return abs(_dressed_energy(bare_b) - _dressed_energy(bare_a)) * 1e9

    def bare_state_labels(self, flux=None, tunable_qubit=None, evals_count=10) -> list:
        """Convenience wrapper: diagonalize (see :meth:`dressed_eigensys`) and
        label each of the ``evals_count`` dressed levels with its dominant
        bare product state via :func:`label_bare_states`.

        Prefer calling :func:`label_bare_states` directly on an
        already-computed ``(evals, evecs, hs)`` (e.g. from a flux sweep) to
        avoid re-diagonalizing for every flux point.
        """
        _evals, evecs, hs = self.dressed_eigensys(
            flux=flux, tunable_qubit=tunable_qubit, evals_count=evals_count,
        )
        dims = [s.truncated_dim for s in hs.subsystem_list]
        return label_bare_states(evecs, dims)


class tunable_coupler(edge):
    """Flux-tunable SQUID-based coupler between two qubits.

    The coupler is itself a transmon-like circuit whose Josephson energy (and
    thus frequency) can be tuned by an external magnetic flux.  At a specific
    flux bias the effective qubit–qubit coupling is suppressed to zero,
    enabling fast high-fidelity two-qubit gates.

    Parameters
    ----------
    q0 : transmon-like
        Control qubit.
    q1 : transmon-like
        Target qubit.
    E_j_max : float
        Maximum Josephson energy of the coupler (Hz).
    E_c : float
        Charging energy of the coupler (Hz).
    C_0c : float
        Coupling capacitance between q0 and the coupler (F).
    C_1c : float
        Coupling capacitance between q1 and the coupler (F).
    d : float, optional
        SQUID junction asymmetry parameter (:math:`0 \\le d < 1`).
        Default 0 (symmetric SQUID).
    flux : float, optional
        Reduced magnetic flux :math:`\\Phi / \\Phi_0`.  Default 0.
    C_12 : float, optional
        Residual direct qubit–qubit coupling capacitance (F).  Default 0.
    truncated_dim : int, optional
        Hilbert space dimension for the coupler qmodel.  Default 4.

    References
    ----------
    [Chen2014] Phys. Rev. Lett. 113, 220502;
    [Yan2018]  Phys. Rev. Applied 10, 054062.
    """

    def __init__(
        self,
        q0,
        q1,
        E_j_max: float,
        E_c: float,
        C_0c: float,
        C_1c: float,
        d: float = 0.0,
        flux: float = 0.0,
        C_12: float = 0.0,
        truncated_dim: int = 4,
    ):
        super().__init__(q0, q1)
        self.E_j_max     = E_j_max
        self.E_c         = E_c
        self.C_0c        = C_0c
        self.C_1c        = C_1c
        self.d           = d
        self.flux        = flux
        self.C_12        = C_12
        self.truncated_dim = truncated_dim

        self.qmodel = scq.TunableTransmon(
            EJmax=E_j_max * 1e-9,
            EC=E_c * 1e-9,
            d=d,
            flux=flux,
            ng=0.0,
            ncut=10,
            truncated_dim=truncated_dim,
        )

    def _f_coupler(self, flux: float) -> float:
        """Coupler 01 transition frequency at a given flux (Hz)."""
        self.qmodel.flux = flux
        return self.qmodel.E01() * 1e9

    def _g_qc(self, qubit, C_qc: float) -> float:
        """Capacitive qubit–coupler coupling at the current coupler frequency."""
        w_c    = 2.0 * pi * self._f_coupler(self.qmodel.flux)
        # Invert Ec = e^2 / (2*C*h)  →  C = e^2 / (2*Ec*h)
        C_c    = e**2 / (2.0 * self.E_c * h)
        w_q    = qubit.omega01()
        C_q    = qubit.C()
        g_rads = 0.5 * np.sqrt(w_q * w_c) * C_qc / np.sqrt(C_q * C_c)
        return g_rads / (2.0 * pi)

    def g(self, flux: Optional[float] = None) -> float:
        r"""Flux-tunable effective qubit–qubit coupling (Hz).

        The effective coupling is the sum of a direct capacitive term and a
        coupler-mediated exchange term:

        .. math::

            g_{\mathrm{eff}}(\Phi) = g_{\mathrm{direct}}
              + g_{0c}(\Phi)\,g_{1c}(\Phi)
                \left(\frac{1}{\Delta_{0c}(\Phi)} + \frac{1}{\Delta_{1c}(\Phi)}\right)

        where :math:`\Delta_{ic}(\Phi) = f_i - f_c(\Phi)`.

        Parameters
        ----------
        flux : float, optional
            Reduced flux :math:`\Phi / \Phi_0`.  If not given, uses
            ``self.flux``.

        Returns
        -------
        float
            Effective coupling strength in Hz.

        References
        ----------
        [Chen2014] Eq. (1); [Yan2018] Eq. (S1).
        """
        if flux is None:
            flux = self.flux
        saved_flux = self.qmodel.flux
        self.qmodel.flux = flux

        f_c   = self._f_coupler(flux)
        g_0c  = self._g_qc(self.q0, self.C_0c)
        g_1c  = self._g_qc(self.q1, self.C_1c)
        Delta_0c = self.q0.f01() - f_c
        Delta_1c = self.q1.f01() - f_c

        if Delta_0c == 0.0 or Delta_1c == 0.0:
            self.qmodel.flux = saved_flux
            raise ValueError("g() is singular: a qubit is resonant with the coupler.")

        g_mediated = g_0c * g_1c * (1.0 / Delta_0c + 1.0 / Delta_1c)
        g_direct   = _g_cap_qq(self.C_12, self.q0, self.q1) if self.C_12 > 0.0 else 0.0

        self.qmodel.flux = saved_flux
        return g_direct + g_mediated

    def flux_for_zero_coupling(
        self,
        flux_min: float = 0.0,
        flux_max: float = 0.5,
    ) -> float:
        """Find the flux bias at which the effective coupling vanishes.

        Uses Brent's root-finding method on :meth:`g`.

        Parameters
        ----------
        flux_min, flux_max : float
            Search interval for the reduced flux :math:`\\Phi / \\Phi_0`.

        Returns
        -------
        float
            Reduced flux at which :math:`g_{\\mathrm{eff}} = 0`.

        Raises
        ------
        ValueError
            If no zero crossing is found in [flux_min, flux_max].
        """
        try:
            return brentq(self.g, flux_min, flux_max)
        except ValueError as exc:
            raise ValueError(
                f"No zero coupling found in [{flux_min}, {flux_max}]. "
                "Adjust flux range or coupler parameters."
            ) from exc

    def hilbert_space(self, flux: Optional[float] = None) -> scq.HilbertSpace:
        """Build 3-body HilbertSpace [q0, coupler, q1].

        Parameters
        ----------
        flux : float, optional
            Flux to set on the coupler before building the space.

        Returns
        -------
        scqubits.HilbertSpace
        """
        self._check_qmodels()
        if flux is not None:
            self.qmodel.flux = flux

        hs = scq.HilbertSpace([self.q0.qmodel, self.qmodel, self.q1.qmodel])
        g_0c = self._g_qc(self.q0, self.C_0c)
        g_1c = self._g_qc(self.q1, self.C_1c)

        hs.add_interaction(
            g_strength=g_0c * 1e-9,
            op1=(self.q0.qmodel.n_operator, self.q0.qmodel),
            op2=(self.qmodel.n_operator, self.qmodel),
            add_hc=True,
        )
        hs.add_interaction(
            g_strength=g_1c * 1e-9,
            op1=(self.qmodel.n_operator, self.qmodel),
            op2=(self.q1.qmodel.n_operator, self.q1.qmodel),
            add_hc=True,
        )
        return hs


class hybrid_coupler(edge):
    """Hybrid coupler for 3-D integrated circuits (vacuum-gap interfaces).

    In multi-chip or flip-chip architectures, qubits on adjacent chips interact
    through a vacuum gap that provides both capacitive coupling (through the gap
    geometry) and inductive coupling (through a shared or proximal loop).  Both
    channels contribute coherently; their relative sign can be constructive or
    destructive depending on the circuit layout.

    Parameters
    ----------
    q0 : transmon-like
        Control qubit (bottom chip).
    q1 : transmon-like
        Target qubit (top chip).
    C_12 : float
        Vacuum-gap coupling capacitance in Farads.
    M : float
        Mutual inductance across the 3-D interface in Henries.
        Set to 0 if purely capacitive.

    Notes
    -----
    The total coupling is

    .. math::

        g = g_{\mathrm{cap}} + g_{\mathrm{ind}}
          = g^{(C)}_{12} + g^{(M)}_{12}

    Both terms use the qubit-specific formulas :func:`_g_cap_qq` and
    :func:`_g_ind_qq` defined in this module.

    References
    ----------
    [Rosenberg2017] npj Quantum Inf. 3, 42 (2017);
    [Blais2021] §4.2 and §2.3.
    """

    def __init__(self, q0, q1, C_12: float, M: float = 0.0):
        super().__init__(q0, q1)
        self.C_12 = C_12
        self.M    = M

    def g_cap(self) -> float:
        """Capacitive contribution to the coupling (Hz).

        References
        ----------
        [Blais2021] Eq. (4.9).
        """
        return _g_cap_qq(self.C_12, self.q0, self.q1)

    def g_ind(self) -> float:
        """Inductive contribution to the coupling (Hz).

        References
        ----------
        [Blais2021] §2.3; [Krantz2019] §2.2.
        """
        return _g_ind_qq(self.M, self.q0, self.q1)

    def g(self) -> float:
        r"""Total hybrid coupling strength (Hz).

        Coherent sum of capacitive and inductive contributions:

        .. math::

            g = g_{\mathrm{cap}} + g_{\mathrm{ind}}

        The sign of :math:`g_{\mathrm{ind}}` relative to
        :math:`g_{\mathrm{cap}}` depends on the geometric orientation of the
        mutual inductance.  By convention both are positive here; flip the sign
        of M to represent the opposing orientation.

        Returns
        -------
        float
            Coupling strength in Hz.

        References
        ----------
        [Rosenberg2017] npj Quantum Inf. 3, 42 (2017).
        """
        return self.g_cap() + self.g_ind()

    def hilbert_space(self) -> scq.HilbertSpace:
        """Build 2-body HilbertSpace with both capacitive and inductive terms.

        Two interaction terms are added:
        * Capacitive: :math:`g_{cap} \\, \\hat{n}_0 \\hat{n}_1`
        * Inductive:  :math:`g_{ind} \\, \\hat{\\varphi}_0 \\hat{\\varphi}_1`

        Returns
        -------
        scqubits.HilbertSpace
        """
        self._check_qmodels()
        hs = scq.HilbertSpace([self.q0.qmodel, self.q1.qmodel])
        g_c = self.g_cap()
        g_i = self.g_ind()

        if g_c != 0.0:
            hs.add_interaction(
                g_strength=g_c * 1e-9,
                op1=(self.q0.qmodel.n_operator, self.q0.qmodel),
                op2=(self.q1.qmodel.n_operator, self.q1.qmodel),
                add_hc=True,
            )
        if g_i != 0.0:
            hs.add_interaction(
                g_strength=g_i * 1e-9,
                op1=(self.q0.qmodel.phi_operator, self.q0.qmodel),
                op2=(self.q1.qmodel.phi_operator, self.q1.qmodel),
                add_hc=True,
            )
        return hs
