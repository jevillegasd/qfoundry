"""Transmon and tunable transmon models.

Class hierarchy
---------------
:class:`qubit`
    Mixin base for any qubit circuit.  Provides zero-point fluctuation (ZPF)
    parameters and other qubit-generic derived quantities.
:class:`transmon` (:class:`qubit`, :class:`~qfoundry.resonator.circuit`)
    Single-junction transmon; inherits both the qubit ZPF interface and the
    RLC circuit model.
:class:`tunable_transmon` (:class:`transmon`)
    SQUID-based flux-tunable transmon.

References
----------
- Koch et al., Phys. Rev. A 76, 042319 (2007)
- Krantz et al., Appl. Phys. Rev. 6, 021318 (2019) / arXiv:1904.06560
"""

from abc import ABC, abstractmethod

from qfoundry.circuit import circuit
from qfoundry.resonator import cpw, cpw_resonator
from qfoundry.utils import sc_metal, Ic_to_R, R_to_Ic
import scqubits as scq

from scipy.constants import e as e_0
from scipy.constants import Planck as h_0, hbar, pi

from numpy import cos, sin, sqrt, tanh, abs, exp, pi 
from numpy import diag, ones, arange, diff, ndarray
from math import factorial

from scipy.linalg import eigh
import inspect


class qubit(ABC):
    r"""Base mixin for qubit circuits.

    Provides zero-point fluctuation (ZPF) parameters and other qubit-generic
    derived quantities.  Subclasses are expected to expose :meth:`Ej`,
    :meth:`Ec`, :meth:`C`, :meth:`L`, and :meth:`omega01`.

    ZPF quantities
    --------------
    The four fundamental ZPF amplitudes for a Josephson circuit in the
    harmonic approximation are:

    .. math::

        n_\mathrm{zpf}   &= \left(\frac{E_J}{8E_C}\right)^{1/4}\frac{1}{\sqrt{2}} \\
        \varphi_\mathrm{zpf} &= \left(\frac{8E_C}{E_J}\right)^{1/4}\frac{1}{\sqrt{2}} \\
        I_\mathrm{zpf}   &= \sqrt{\frac{\hbar\,\omega_{01}}{2L_J}} \\
        V_\mathrm{zpf}   &= \sqrt{\frac{\hbar\,\omega_{01}}{2C_\Sigma}}

    Note that :math:`n_\mathrm{zpf}\cdot\varphi_\mathrm{zpf} = 1/2`, consistent
    with the canonical commutation relation
    :math:`[\hat{\varphi},\hat{n}] = i`.

    References
    ----------
    - Koch et al., Phys. Rev. A 76, 042319 (2007), Eq. (2.7)
    - Krantz et al., Appl. Phys. Rev. 6, 021318 (2019), Eq. (17-18)
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def Ej(self) -> float:
        r"""Josephson energy in Hz (i.e. :math:`E_J / h`).  Subclasses must override.

        The Josephson energy sets the tunnelling energy of the junction:

        .. math::

            E_J/h = \frac{I_c}{4 \, \pi \, e} = \frac{I_c}{2\, e}

        References
        ----------
        Koch et al. (2007) Eq. (2.2).
        """


    @abstractmethod
    def Ec(self) -> float:
        r"""Charging energy in Hz (i.e. :math:`E_C / h`).  Subclasses must override.

        .. math::

            E_C/h = \frac{e^2}{2\,h \, C_\Sigma}
            

        References
        ----------
        Koch et al. (2007) Eq. (2.3).
        """

    def El(self, phi=0.0) -> float:
        r"""Inductive energy in Hz (i.e. :math:`E_L / h`).

        For inductances different from the Josephson inductance, the inductive energy is

        .. math::

            E_L/h = \frac{h}{8 \, e^2 \, L}

        where :math:`L` is the inductance.  Returns 0 by default (no geometric
        inductance); subclasses may override.
        """
        return 0

    # ------------------------------------------------------------------
    # Derived circuit quantities — computed from Ej and Ec
    # ------------------------------------------------------------------

    def Ic(self) -> float:
        r"""Critical current of the Josephson junction (A).

        .. math::

            :math:`I_c = E_J \cdot 4\pi e` (with :math:`E_J` in Hz units).

        Returns
        -------
        float
            Critical current in Amperes.

        References
        ----------
        Koch et al. (2007) Eq. (2.2).
        """
        return self.Ej() * 4.0 * pi * e_0

    def L_J(self, phi=0.0) -> float:
        r"""Josephson inductance (H).

        .. math::

            L_J(\phi) = \frac{\hbar}{2\,e\,I_c\,\cos\phi}

        Parameters
        ----------
        phi : float
            Reduced flux bias in radians.  Default ``0.0``.
        """
        from numpy import cos
        return hbar / (2 * e_0 * self.Ic()) / cos(phi)

    def I_zpf(self) -> float:
        r"""Zero-point current fluctuation (A).

        .. math::

            I_\mathrm{zpf} = \sqrt{\frac{\hbar\,\omega_{01}}{2\,L_J}}

        where :math:`L_J` is the Josephson inductance at the qubit operating
        point.

        Returns
        -------
        float
            :math:`I_\mathrm{zpf}` in Amperes.

        References
        ----------
        Koch et al. (2007) Eq. (2.7); Krantz et al. (2019) Eq. (18).
        """
        return sqrt(hbar * self.omega01() / (2.0 * self.L_J()))

    def V_zpf(self) -> float:
        r"""Zero-point voltage fluctuation (V).

        .. math::

            V_\mathrm{zpf} = \sqrt{\frac{\hbar\,\omega_{01}}{2\,C_\Sigma}}

        Returns
        -------
        float
            :math:`V_\mathrm{zpf}` in Volts.

        References
        ----------
        Krantz et al. (2019) Eq. (17).
        """
        C = e_0**2 / (2 * self.Ec() * h_0)  # Total qubit capacitance from Ec
        return sqrt(hbar * self.omega01() / (2.0 * C))

    def phi_zpf(self) -> float:
        r"""Zero-point phase (flux) fluctuation (dimensionless, reduced flux units).

        In the harmonic approximation of the cosine potential:

        .. math::

            \varphi_\mathrm{zpf} = \left(\frac{8\,E_C}{E_J}\right)^{1/4}
                                   \frac{1}{\sqrt{2}}

        Returns
        -------
        float
            :math:`\varphi_\mathrm{zpf}` (dimensionless).

        References
        ----------
        Koch et al. (2007) Eq. (2.7).
        """
        return (8.0 * self.Ec() / self.Ej()) ** 0.25 / sqrt(2.0)

    def n_zpf(self) -> float:
        r"""Zero-point charge fluctuation (in units of Cooper pairs, i.e., 2e).

        In the harmonic approximation:

        .. math::

            n_\mathrm{zpf} = \left(\frac{E_J}{8\,E_C}\right)^{1/4}
                             \frac{1}{\sqrt{2}}

        This equals the charge matrix element
        :math:`|\langle 0 |\hat{n}| 1 \rangle|` in the transmon regime
        (see :meth:`transmon.n01`).

        Returns
        -------
        float
            :math:`n_\mathrm{zpf}` (dimensionless).

        References
        ----------
        Koch et al. (2007) Eq. (2.7).
        """
        return (self.Ej() / (8.0 * self.Ec())) ** 0.25 / sqrt(2.0)

    
    

class transmon(qubit, circuit):
    """
    Single Junction Qubit
        E_j:    float               # Josephson energy
        C_sigma:  float=67.5e-15,
        C_g:    float  =21.7e-15,
        C_k:    float  =36.7e-15,
        C_d:   float =0.0e-15,
        res_ro = cpw_resonator(cpw(11.7,550 ,15,7.5, 0.2, alpha=2.4e-2),frequency = 7e9, length_f = 2),    #Readout Resonator
        R_jx:   float = 0.0,       # Resistance correction factor
        mat =   sc_metal(1.14),
        T =     20.e-3,
        kappa = 0.0,
        ng = 0.3 #Offset Charge
        NOTE: Energies are in E/h (not E/hbar)
    """
    qmodel = None

    def __init__(
        self,
        E_j: float,  # Josephson energy
        E_c: float = None,
        C_g: float = None,   # Coupling capacitance (overrides g when provided)
        g: float = None,     # Coupling strength between qubit and resonator (Hz)
        C_d: float = 0.0e-15, # Capacitance to drive
        res_ro=cpw_resonator(
            cpw(11.7, 550, 15, 7.5, 0.2, alpha=2.4e-2), frequency=7e9, length_f=4 # Readout Resonator
        ),  
        mat=sc_metal(1.14, 20e-3),
        inst_model = True,  # Whether to instantiate the scquibits model
        **kwargs
    ):
        self.mat = mat
        self._ej_ = E_j
        self._ec_ = E_c   

        if E_c is None:
            C_sigma = kwargs.get("C_sigma", None)
            assert C_sigma is not None, "Either E_c or C_sigma should be provided."
            self._ec_ = e_0**2 / (2 * C_sigma) / h_0

        # Junction resistances are Josephson-junction fabrication parameters used to
        # derive the critical current (Ambegaokar-Baratoff relation).  They are *not*
        # classical dissipation resistances and must not be confused with circuit._R_.
        #   _Rj_  — normal-state junction resistance (from measurement or target spec)
        #   _Rx_  — series parasitic resistance correction (e.g. contact/lead resistance)
        self._Rx_ = kwargs.get("R_jx", 0) or 0
        self._Rj_ = kwargs.get("R_j", 0) or 0

        self.C_d = C_d
        self.res_ro = res_ro
        self.kappa = kwargs.get("kappa", self.res_ro.kappa_ext()) # External coupling rate

        self._g_ = g
        # Derive C_g from g if not explicitly provided.
        # Inverts: g = 2 * (C_g / C_sigma) * e_0 * V_zpf * n01 / h_0
        if C_g is None and g is not None:
            C_sigma = e_0**2 / (2 * self._ec_ * h_0)
            V_zpf = self.res_ro.V_zpf()
            n01 = (self._ej_ / (8.0 * self._ec_)) ** 0.25 / sqrt(2.0)
            C_g = g * h_0 * C_sigma / (2 * e_0 * V_zpf * n01)
        self.C_g = C_g

        # scqubits parameters
        self.ng = kwargs.get("ng", 0.3)     # Offset charge
        self.ncut = kwargs.get("ncut", 8)  # Number of states to truncate the Hilbert space
        self.truncated_dim = kwargs.get("truncated_dim", 8) # Number of states to truncate the Hilbert space

        # Instantiate the scqubits model
        if inst_model:
            self.qmodel = scq.Transmon(
                EJ=self.Ej() / 1e9,
                EC=self.Ec() / 1e9,
                ng=self.ng,
                ncut=self.ncut,
                truncated_dim=self.truncated_dim,
            )
 
        self._R_ = kwargs.get("R_loss", float('inf'))

    @classmethod
    def from_ic(cls, i_c: float, R_jx: float = 0.0, **kwargs):
        """Initialize transmon from critical current :math:`I_c`.

        Derives the Josephson energy :math:`E_J = I_c / (4 e \\pi)` and
        estimates the normal-state junction resistance via the
        Ambegaokar–Baratoff relation.

        Parameters
        ----------
        i_c : float
            Critical current in Amperes.
        R_jx : float, optional
            Series parasitic resistance correction (e.g. contact/lead
            resistance) in Ohms.  The stored junction resistance is
            ``R_j = R_AB - R_jx``.  Default is ``0.0``.
        **kwargs
            Additional keyword arguments passed to :meth:`__init__`
            (e.g. ``E_c``, ``C_sigma``, ``C_g``, ``res_ro``, ``mat``).

        Returns
        -------
        transmon
        """
        E_j = i_c / (4 * e_0 * pi)
        Rj = Ic_to_R(i_c, mat=kwargs.get("mat", sc_metal(1.14, 25e-3))) - R_jx
        kwargs["R_j"] = Rj
        kwargs["R_jx"] = R_jx
        return cls(E_j=E_j, **kwargs)

    @classmethod
    def from_rj(cls, R_j: float, R_jx: float = 0.0, **kwargs):
        """Initialize transmon from normal-state junction resistance :math:`R_J`.

        Derives the critical current via the Ambegaokar–Baratoff relation using
        the effective resistance ``R_j + R_jx``, then computes
        :math:`E_J = I_c / (4 e \\pi)`.

        Parameters
        ----------
        R_j : float
            Measured normal-state junction resistance in Ohms.
        R_jx : float, optional
            Series parasitic resistance correction in Ohms.  The effective
            resistance used for the AB relation is ``R_j + R_jx``.  Default
            is ``0.0``.
        **kwargs
            Additional keyword arguments passed to :meth:`__init__`.
            Must include either ``E_c`` or ``C_sigma``.

        Returns
        -------
        transmon
        """
        from numpy import isnan
        if R_jx is None or isnan(R_jx):
            R_jx = 0.0

        mat = kwargs.get("mat", sc_metal(1.14, 25e-3))

        E_c = kwargs.get("E_c", None)
        if E_c is None:
            C_sigma = kwargs.get("C_sigma", None)
            assert C_sigma is not None, "C_sigma must be provided if E_c is not."
            E_c = e_0**2 / (2 * C_sigma) / h_0

        i_c = R_to_Ic(R_j + R_jx, mat=mat)
        E_j = i_c / (2 * e_0 * 2 * pi)
        kwargs["R_j"] = R_j
        kwargs["R_jx"] = R_jx
        kwargs["E_c"] = E_c
        kwargs["E_j"] = E_j
        kwargs.pop("C_sigma", None)   # already consumed above; not a transmon.__init__ param
        return cls(**kwargs)

    @classmethod
    def from_f01(cls, f01: float, R_jx: float = 0.0, **kwargs):
        """Initialize transmon from a target qubit frequency :math:`f_{01}`.

        Inverts the leading-order approximation
        :math:`f_{01} \\approx \\sqrt{8 E_J E_C} - E_C` (Koch 2007, Eq. 3.1)
        to find the required :math:`E_J` for a given :math:`E_C`, then
        estimates the junction resistance via the Ambegaokar–Baratoff relation.

        Parameters
        ----------
        f01 : float
            Target qubit transition frequency in Hz.
        R_jx : float, optional
            Series parasitic resistance correction in Ohms.  Subtracted from
            the AB-derived resistance to obtain the stored ``R_j``.  Default
            is ``0.0``.
        **kwargs
            Additional keyword arguments passed to :meth:`__init__`.
            Must include either ``E_c`` or ``C_sigma``.
            If ``R_j`` is provided, the junction resistance is not recomputed.

        Returns
        -------
        transmon
        """
        Rj = kwargs.get("R_j", None)

        E_c = kwargs.get("E_c", None)
        if E_c is None:
            C_sigma = kwargs.get("C_sigma", None)
            assert C_sigma is not None, "Either E_c or C_sigma must be provided."
            E_c = e_0**2 / (2 * C_sigma) / h_0

        # Use the resolved E_c (whether passed directly or derived from C_sigma)
        ec = E_c

        # Invert f01 = sqrt(8*Ej*Ec) - Ec  (Koch 2007 Eq. 3.1, leading order)
        # TODO: Replace with numerical Mathieu/Hamiltonian solver for higher accuracy.
        ej = (f01 + ec) ** 2 / (8 * ec)
        ic = ej * 4 * e_0 * pi

        if Rj is None:  # The resistance value is only calculated if not provided
            Rj = Ic_to_R(ic, mat=kwargs.get("mat", sc_metal(1.14, 25e-3))) - R_jx

        kwargs["R_j"] = Rj
        kwargs["R_jx"] = R_jx
        kwargs.pop("E_c", None)       # passing E_c explicitly
        kwargs.pop("C_sigma", None)   # already consumed above; not a transmon.__init__ param
        return cls(E_j=ej, E_c=ec, **kwargs)

    def alpha(self):
        """
        Anharmonicity
        """
        from numpy.linalg import LinAlgError 
        from numpy import sqrt
        # Analytical approximation for anharmonicity, in Hz
        alpha = -self.Ec()* (1-1/4*sqrt(self.Ec()/(8*self.Ej()))) # https://arxiv.org/pdf/cond-mat/0703002 eq. 3.1 and eq. 3.4
        if self.qmodel is not None:
            try: 
                return self.qmodel.anharmonicity() * 1e9
            except LinAlgError as e:
                return alpha
        return alpha

    def L(self, phi=0.0):
        r"""Circuit inductance for the RLC model (H).

        For a plain transmon there is no geometric inductance; this delegates
        to :meth:`L_J` so that circuit-level methods (``_f0_``, ``Q``, etc.)
        see the Josephson inductance at the operating point.

        Parameters
        ----------
        phi : float
            Reduced flux bias passed through to :meth:`L_J`.  Default ``0.0``.
        """
        return self.L_J(phi)

    def C(self):
        """
        RLC circuit model capacitance for the ground state

        .. math::
            C_\Sigma = \frac{e^2}{2 E_C h}
        """
        return e_0**2 / (2 * self.Ec() * h_0)

    def Ic(self):
        return self._ej_ * 2 * e_0 * (2 * pi)

    def Ec(self):
        """
        Capacitive energy
        """
        return self._ec_

    def Ej(self):
        """
        Josephson energy
        """
        return self._ej_

    def nj(self,j):
        """
        Charge matrix element between states j+1 and j
        .. math::
            |< j + 1 | \hat{n} | j >| \approx \sqrt{\frac{1+j}{2}}\left(\frac{E_j}{8 E_c}\right)^{1/4}

        Parameters
        ----------
        j: int
            State index (0 for n01, 1 for n12, etc.)
        https://arxiv.org/pdf/cond-mat/0703002 eq 3.4
        """
        n_j_annalytical = sqrt((1+j)/2)*(self.Ej() / (8*self.Ec()))**(1/4)
        return n_j_annalytical
    
    def g_j(self,j):
        """
        Coupling sytrength between states j and j+1
        """
        if self.C_g is None:
            # Scale g01 by the ratio of charge matrix elements
            return self.g() / 2 * self.nj(j) / self.n01()

        nj = self.nj(j)

        beta    = self.C_g / self.C() # Participation ratio
        V_zpf    = self.res_ro.V_zpf() # Resonator zero-point voltage fluctuation

        g_j = beta * V_zpf * nj * e_0/h_0
        return g_j
    
    def g01(self):
        return self.g()
    
    def g(self):
        """
        Coupling strength between the qubit and the resonator (capacitive)
        https://arxiv.org/pdf/cond-mat/0703002 eq. 3.3

        g01 ~ 2 * beta * e * Vrms * n01 / h
        where beta is the participation ratio (beta ~= C_g/ (C_sigma)) [coupling capaictance over total capacitance]
        e is the elementary charge,
        Vrms is the root mean square voltage across the resonator, and hbar is the reduced Planck's constant.
        The Vrms can be calculated from the resonator frequency and its capacitance as
        V_rms = sqrt(hbar * omega_r / (2 * C_r))
        
        https://arxiv.org/pdf/cond-mat/0703002 eq. 3.1*

        Another option is to use the numerical charge matrix element n_matrix()[0,1]
        g = C_g*(2*Ec/e) * V_zpf * n_matrix()[0,1] / hbar

        """

        if self.C_g is None:
            return self._g_

        C_sigma = self.C()
        f_r     = self.res_ro.f0()      # Resonator frequency
        C_r     = self.res_ro.C()       # Resonator capacitance (used in fallback)
        V_zpf    = self.res_ro.V_zpf()  # Resonator zero-point voltage fluctuation
        C_g   = self.C_g
        try:
            assert self.n01() is not None
        except:
            return C_g/2*sqrt(f_r*self.f01()/ (C_sigma*C_r))  # Fallback if n01 is not available
        
        beta    = self.C_g / C_sigma # Participation ratio
        return 2* beta * e_0 * V_zpf * self.n01() / h_0  # in Hz
        # return 2 * beta * e_0 * V_zpf * self.n_matrix()[0,1] / h_0 # Numerical alternative for n01

    def n01(self):
        """
        Estimated first charge matrix element between the ground and first excited state
        """
        n01 = (self.Ej()/(8*self.Ec()))**0.25 * 1/ sqrt(2)  # https://arxiv.org/pdf/cond-mat/0703002 eq 3.4
        return n01
    
    def n_matrix(self, N=20, nlevels=10):
        """
        Charge matrix of the qubit
        N: ncharge basis truncation
        """
        ec = self.Ec()
        ej = self.Ej()
        n = arange(-N, N+1)
        H = diag(4 * ec * n**2) - 0.5 * ej * (diag(ones(2*N), 1) + diag(ones(2*N), -1))
        eigvals, eigvecs = eigh(H)
        n_op = diag(n)
        n_matrix = eigvecs.T @ n_op @ eigvecs

        return n_matrix[:nlevels, :nlevels], eigvals[:nlevels] # Energies same units as Ec and Ej (Hz)

    def chi(self):
        """
        Dispersive shift
        https://arxiv.org/pdf/1904.06560 eq. 146
        """
        return -(self.g01() ** 2) / (self.delta()) * (1 / (1 + self.delta() / self.alpha()))

    def E_m(self, m):
        """
        Energy of the m-th level
        """
        from numpy.linalg import LinAlgError
        if self.qmodel is not None: 
            try:
                spectrum = self.qmodel.get_spectrum_vs_paramvals(param_name = "ng",  param_vals = [0, 0.5])
                E_m = spectrum.energy_table[m,0] # Energy at ng=0
                return E_m * 1e9
            except LinAlgError as e:
                return -self.Ej() + sqrt(8*self.Ej()*self.Ec())*(m+0.5) - self.Ec()*(6*m**2 + 6*m + 3)/12
            
        else:
            return -self.Ej() + sqrt(8*self.Ej()*self.Ec())*(m+0.5) - self.Ec()*(6*m**2 + 6*m + 3)/12
    

    def E01(self):
        """01 transition energy in Hz.  E01 = E_1 - E_0 ≈ sqrt(8 Ej Ec) - Ec."""
        from numpy.linalg import LinAlgError
        if self.qmodel is not None:
            try:
                return self.qmodel.E01() * 1e9
            except LinAlgError:
                pass  # fall through to analytical approximation
        # Koch et al. (2007) Eq. (3.1): E01 = sqrt(8*Ej*Ec) - Ec
        return sqrt(8 * self.Ej() * self.Ec()) - self.Ec()


    def f01(self):
        """
        Qubit 01 frequency, in Hz (equivalent to E01)
        """
        return self.E01()
       

    def omega01(self):
        """
        Qubit 01 angular frequency
        """
        return 2 * pi * self.f01()

    def f12(self):
        """
        Qubit 12 frequency
        """
        return self.f01() + self.alpha()

    def f02(self):
        """
        Qubit 02/2 frequency
        """
        return self.f01() + self.f12()

    def SNR(self, T1=100e-6):
        """
        SNR
        """
        gamma_1 = 1 / T1
        return (
            self.kappa
            * self.chi() ** 2
            / (gamma_1 * (self.kappa**2 / 4 + self.chi() ** 2))
        )
    
    def delta(self):
        """
        Frequency detuning
        """
        return abs(self.res_ro.f0() - self.f01())

    def T1_max(self):
        """
        Higher bound of T1 (Purcell limit)
        https://arxiv.org/pdf/cond-mat/0703002 eq 4.7
        """
        return (self.delta() / self.g01()) ** 2 / (self.res_ro.kappa_ext())
    
    def T1_drive(self, Z0=50):
        """
        T1 limited by the drive coupling
        """
        from numpy import floating, ndarray
        C_c = self.C_d

        # Check if C_c is a positive float or a list/array of two elements
        if not (isinstance(C_c, (float, floating)) and C_c > 0) and not (isinstance(C_c, (list, ndarray)) and len(C_c) == 2):
            print("C_d should be a positive float or a list/array of two elements.")
            return float('inf')
        else:
            if isinstance(C_c, (list, ndarray)):
                assert len(C_c) == 2, "C_d must be a list/array of two elements."
                return self.C() / (Z0 * (self.omega01())**2 * (diff(C_c)[0]**2)) / (2*pi) # * 4 # Double check this derivation
            return self.C() / (Z0 * (self.omega01())**2 * (C_c)**2) / (2*pi)

    def omega_rabi(self, V_rms):
        """
        Rabi frequency under a drive with root mean square voltage V_rms
        """
        from numpy import floating
        C_sigma = self.C()
        C_c = self.C_d if isinstance(self.C_d, (floating, float)) else abs(diff(self.C_d)[0])

        try:
            n01 = self.n01()
        except:
            n01 = 1.2

        beta = C_c / C_sigma  # Participation ratio
        return (2 * e_0 * n01 * beta * V_rms)/hbar # Rabi frequency
    
    def tau_rabi(self, P_in=-63, Z0=50):
        """
        Rabi time under a drive with root mean square voltage V_rms
        Arguments:
        P_in : float
            Input power in Watts (if negative, assumed to be in dBm)
        Z0 : float
            Characteristic impedance of the drive line (default: 50 Ohm)

        """
        if P_in <= 0:
            # Power is probably in dB
            P_in = 10 ** (P_in / 10) * 1e-3  # Convert dBm to Watts

        V_rms = sqrt(2 * Z0 * P_in)  # Root mean square voltage
        tau_pi = pi / self.omega_rabi(V_rms)  # Time for a pi pulse
        return tau_pi
    
    def epsilon_m(self, m = 0):
        """
        Charge dispersion for the m-th level
        https://arxiv.org/pdf/cond-mat/0703002 eq 3.5
        """
        from numpy.linalg import LinAlgError

        eps_m = (-1)**m * self.Ec() * (2**(4*m+5) / factorial(m)) * sqrt(2/pi) * (self.Ej()/(8*self.Ec()))**(m/2 + 3/4) * exp(-sqrt(8*self.Ej()/self.Ec())) # Analytical approximation for charge dispersion, valid in the transmon regime (EJ/EC >> 1)
        if self.qmodel is not None:
            try:
                spectrum = self.qmodel.get_spectrum_vs_paramvals(param_name = "ng",  param_vals = [0, 0.5])
                epsilon_m = abs(spectrum.energy_table[m,1] - spectrum.energy_table[m,0])
                return epsilon_m * 1e9
            except LinAlgError as e:
                return eps_m
        else:
            return eps_m


    def __str__(self):
        return (
            "Ec = \t%3.2f MHz \nEj = \t%3.2f GHz \nIc = \t%3.2f nA \nEJ/EC= \t%1.2f\nf_01 = \t%3.2f GHz \n"
            "f_02 = \t%3.2f GHz \ng_01 = \t%3.2f MHz \nchi =\t%3.2f MHz \nT1_max =\t%3.2f us\n"
            "alpha =\t%3.2f MHz"
            % (
                self._ec_ * 1e-6,
                self._ej_ * 1e-9,
                self.Ic() * 1e9,
                self._ej_ / self._ec_,
                self.f01() * 1e-9,
                self.f02() * 1e-9,
                self.g01() * 1e-6,
                self.chi() * 1e-6,
                self.T1_max() * 1e6,
                self.alpha() * 1e-6,
            )
        )
    
    # print
    def __repr__(self):
        return super().__repr__() + self.__str__()


class tunable_transmon(transmon):
    """
    Tunable Transmon Qubit
    """
    flux = 0
    def __init__(self, flux=0.0, d=1, *args, **kwargs):
        self.flux = flux
        self.d = d  # Assymetry parameter
        self.ng = kwargs.get("ng", 0.3)  # Offset charge
        super().__init__(inst_model=False, *args, **kwargs)

        # Check relevant args for TunableTransmon available in kwargs
        for key, value in kwargs.items():
            if key in [
                "ncut",
                "truncated_dim",
            ]:
                setattr(self, key, value)

        self.qmodel = scq.TunableTransmon(
            EJmax=self.Ej() / 1e9,
            EC=self.Ec() / 1e9,
            d=self.d,
            flux=self.flux,
            ng=self.ng,
            ncut=self.ncut,
            truncated_dim=self.truncated_dim,
        )

    @classmethod # Overrides the parent class method (but is doing the same)
    def from_ic(cls, i_c: float, **kwargs):
        """
        Initialize tunable transmon from critical current I_c.
        """
        E_j = i_c / (4 * e_0 * pi)
        Rx = kwargs.get("R_jx", 0.0)
        Rj = Ic_to_R(i_c, mat=kwargs.get("mat", sc_metal(1.14, 25e-3))) - Rx
        kwargs["R_j"] = Rj
        kwargs["R_jx"] = Rx
        return cls(E_j=E_j, **kwargs)
    
    def Ej1(self):
        """
        Josephson energy for the first junction
        """
        return self._ej_ * (1 + self.d) / 2

    def Ej2(self):
        """
        Josephson energy for the second junction
        """
        return self._ej_ * (1 - self.d) / 2

    def Ic1(self):
        """
        Critical current for the first junction
        """
        return self.Ic() * (1 + self.d) / 2

    def Ic2(self):
        """
        Critical current for the second junction
        """
        return self.Ic() * (1 - self.d) / 2
    
    def f01(self, flux=None):
        """
        Qubit 01 frequency
        """
        if flux is None:
            flux = [self.flux]
        elif isinstance(flux, (int, float)):
            flux = [flux]
        if self.qmodel is not None:
            # By default return the f01 using scqubits
            spectrum = self.qmodel.get_spectrum_vs_paramvals(param_name = "flux",  param_vals = flux)
            f01 = spectrum.energy_table[:,1]-spectrum.energy_table[:,0]
            return f01 * 1e9 if len(f01) > 1 else f01[0] * 1e9
        else:
            # Effective Ej for asymmetric SQUID (Koch 2007 / Krantz 2019)
            Ej_eff = self.Ej() * sqrt(cos(pi * flux) ** 2 + self.d**2 * sin(pi * flux) ** 2)
            # Koch et al. (2007) Eq. (3.1): f01 ≈ sqrt(8*Ej_eff*Ec) - Ec
            return sqrt(8 * Ej_eff * self.Ec()) - self.Ec()

    def alpha(self, flux=None):
        """
        Anharmonicity at a given flux bias (reduced flux, Phi/Phi_0).

        Mirrors f01(flux=...): with no argument, behaves exactly like the
        base transmon.alpha() (evaluated at self.qmodel's current flux).
        With an explicit flux, temporarily biases self.qmodel to compute the
        anharmonicity there, then restores the original bias.
        """
        if flux is None:
            return super().alpha()
        saved_flux = self.qmodel.flux
        self.qmodel.flux = flux
        try:
            return super().alpha()
        finally:
            self.qmodel.flux = saved_flux


    def __str__(self):
        return super().__str__() + "\nFlux = \t%3.2f" % self.flux
