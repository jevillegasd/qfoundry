"""Resonator and CPW utilities.

This module implements coplanar waveguide (CPW) resonators based on the distributed
circuit model from Wallraff et al. (2008). The resonator is modeled as a lumped
LC circuit with parameters derived from the CPW geometry and material properties.

References
----------
- Ghione (1984), doi:10.1049/el:19840120 - CPW impedance and capacitance calculations
- Watanabe (1994), doi:10.1143/JJAP.33.5708 - CPW effective permittivity
- Wallraff et al. (2008), arXiv:0807.4094 - Main reference for CPW resonator theory
"""

import warnings
from scipy.constants import c, epsilon_0, m_e, hbar, e, k, pi, Avogadro
from scipy.optimize import fsolve
import numpy as np
import scqubits as scq

from qfoundry.waveguides import cpw
from qfoundry.circuit import circuit  # re-exported for backward compatibility


class cpw_resonator(circuit):
    """
    A coplanar waveguide resonator based on Wallraff et al. (2008).
    
    Implements the distributed LC model where the resonator is treated as a
    lumped circuit with effective L and C values derived from the CPW geometry.
    
    The resonator frequency is given by f₀ = 1/(2π√LC) where:
    - L = 2*L_l*l/(n*π)² (half-wave) or 8*L_l*l/((2n-1)*π)² (quarter-wave)
    - C = C_l*l/2 + C_c 
    
    Parameters
    ----------
    wg : cpw
        The coplanar waveguide defining the transmission line properties
    frequency : float
        Target resonance frequency in Hz
    length_f : int, default=2
        Length factor: 2 for half-wave, 4 for quarter-wave resonator
    n : int, default=1
        Mode number (fundamental = 1)
    Cg : float, default=0.0
        Coupling capacitance to ground in F
    Ck : float, default=0.0
        Coupling capacitance to feedline in F
    R_L : float, default=50.0
        Load resistance in Ohms
        
    Attributes
    ----------
    length : float
        Physical length of the resonator in m
    Cp : float
        Effective coupling capacitance accounting for load impedance
    qmodel : scqubits.Oscillator
        Quantum model of the resonator for multi-level calculations
        
    Notes
    -----
    The coupling capacitance Cp includes the effect of finite load resistance
    following Wallraff Eq. (15). For weak coupling (ωC_k*R_L << 1), Cp ≈ C_k.
    """
    length = None  # Length of the resonator in m
    qmodel = None  # scqubits model of the resonator
    truncated_dim = None
    
    def __init__(
        self,
        wg: cpw,
        frequency: float,
        length_f: int = 4,
        n: int = 1,
        Cg: float = 0.0,
        Ck: float = 0.0,
        R_L: float = 50.0, 
        **kwargs
    ):
        self.wg = wg
        
        if frequency is None:
            raise ValueError("Frequency must be provided, or object instance must be created with a length using from_length method.")

        if int(length_f) not in [1, 2, 4]:
            raise ValueError("length_f must be 1 (full-wave), 2 (half-wave) or 4 (quarter-wave)")
        self.length_f = length_f 
        self.n = n 
        self.Ck = Ck

        wn = 2 * np.pi * frequency * n 

        # Assign coupling factors based on voltage antinodes
        Cp_factor = 2 if length_f in [1, 2] else 1
            
        C_k = Ck + Cg

        # Effective coupling capacitance including load impedance effect
        self.Cp = C_k / (1 + wn**2 * C_k**2 * R_L**2) 
        
        if self.length is None:
            self.length = self._get_length_(
                frequency, self.Cp * Cp_factor, n=n
            ) / self.length_f

        # Effective Capacitance (bare mode C is always half the physical C)
        self._C_ = (self.wg.C_m * self.length) / 2.0 + (self.Cp * Cp_factor)

        # Effective Inductance
        if self.length_f in [1, 2]:
            self._L_ = (2 * self.wg.L * self.length) / (self.n * np.pi)**2
        else:
            self._L_ = (8 * self.wg.L * self.length) / ((2 * self.n - 1) * np.pi)**2

        # Resistance and Losses
        self._R_ = wg.Z_0k / (self.wg.alpha * self.length)
        self._R_ += (1 + wn**2 * C_k**2 * R_L**2) / (
            wn**2 * (C_k + 1e-20)**2 * R_L
        )

        # Quantum Model Initialization
        if self.qmodel is None and kwargs.get("inst_model", True):
            self.truncated_dim = kwargs.get("truncated_dim", 4)
            self.qmodel = scq.Oscillator(
                E_osc=self.f0() * 1e-9, 
                l_osc=self.length,
                truncated_dim=self.truncated_dim, 
            )
            
    @classmethod
    def from_length(cls, length: float, **kwargs):
        """
        Create a resonator from a given length, iteratively solving for frequency 
        to account for frequency-dependent coupling capacitance (Cp).
        """
        kwargs.setdefault("wg", cpw(11.45, 550, 15, 7.5, 0.2, material=kwargs.get("material")))
        wg = kwargs["wg"]
        length_f = kwargs.get("length_f", 2)
        n = kwargs.get("n", 1)
        Ck = kwargs.get("Ck", 0.0)
        Cg = kwargs.get("Cg", 0.0)
        R_L = kwargs.get("R_L", 50.0)

        Cp_factor = 2 if length_f in [1, 2] else 1
        C_k = Ck + Cg

        # Define nominal L and C for the initial frequency guess
        C = (wg.C_m * length) / 2.0 + C_k * Cp_factor
        if length_f in [1, 2]:
            L = (2 * wg.L * length) / (n * np.pi)**2
        else:
            L = (8 * wg.L * length) / ((2 * n - 1) * np.pi)**2

        f0_guess = 1 / (2 * np.pi * np.sqrt(L * C))

        # Use simple approximation for very weak coupling
        if C_k < 1e-15:  
            warnings.warn("Using weak coupling approximation for resonator frequency.")
            return cls(frequency=f0_guess, **kwargs)

        # Iterative solver for strong coupling cases
        max_iterations = 50
        tolerance = 1e-7 

        for iteration in range(max_iterations):
            wn = 2 * np.pi * f0_guess * n
            Cp = C_k / (1 + wn**2 * C_k**2 * R_L**2)
            
            C = (wg.C_m * length) / 2.0 + Cp * Cp_factor
            
            f0_new = 1 / (2 * np.pi * np.sqrt(L * C))
            relative_error = abs(f0_new - f0_guess) / f0_guess
            
            if relative_error < tolerance:
                break
            f0_guess = f0_new
            
        if iteration == max_iterations - 1:
            warnings.warn(f"Frequency iteration did not converge after {max_iterations} iterations. "
                          f"Final relative error: {relative_error:.2e}")
        
        return cls(frequency=f0_new, **kwargs)
    
    @classmethod
    def from_length_exact(cls, length: float, **kwargs):
        """
        Create a resonator from a given length using scipy.optimize.fsolve to
        robustly handle implicit frequency dependence of Cp.
        """
        kwargs.setdefault("wg", cpw(11.45, 550, 15, 7.5, 0.2, material=kwargs.get("material")))
        wg = kwargs["wg"]
        length_f = kwargs.get("length_f", 2)
        n = kwargs.get("n", 1)
        Ck = kwargs.get("Ck", 0.0)
        Cg = kwargs.get("Cg", 0.0)
        R_L = kwargs.get("R_L", 50.0)

        Cp_factor = 2 if length_f in [1, 2] else 1
        C_coupling = Ck + Cg

        def _L(length):
            if length_f in [1, 2]:
                return (2 * wg.L * length) / (n * np.pi) ** 2
            return (8 * wg.L * length) / ((2 * n - 1) * np.pi) ** 2

        # Fast path for weak coupling
        if C_coupling < 1e-15:  
            C = wg.C_m * length / 2
            f0 = 1 / (2 * np.pi * np.sqrt(_L(length) * C))
            return cls(frequency=f0, **kwargs)

        def frequency_equation(f0):
            wn = 2 * np.pi * f0 * n
            Cp = C_coupling / (1 + wn**2 * C_coupling**2 * R_L**2)
            C = wg.C_m * length / 2 + Cp * Cp_factor
            f0_calculated = 1 / (2 * np.pi * np.sqrt(_L(length) * C))
            return f0 - f0_calculated

        # Seed the solver with the weak-coupling approximation
        f0_guess = 1 / (2 * np.pi * np.sqrt(
            _L(length) * (wg.C_m * length / 2 + C_coupling * Cp_factor)
        ))

        try:
            f0_solution = fsolve(frequency_equation, f0_guess, xtol=1e-12)[0]
        except Exception as e:
            warnings.warn(f"Root finding failed: {e}. Using iterative method as fallback.")
            return cls.from_length(length, **kwargs)
        
        return cls(frequency=f0_solution, **kwargs)

    @classmethod
    def from_total_capacitance(cls, C_total: float, **kwargs):
        """
        Create a resonator from a total self-capacitance, e.g. extracted from
        a FEM capacitance-matrix simulation, instead of a target frequency.
        """
        kwargs.setdefault("wg", cpw(11.45, 550, 15, 7.5, 0.2, material=kwargs.get("material")))
        length_f = kwargs.get("length_f", 2)
        Cg = kwargs.get("Cg", 0.0)
        Ck = kwargs.get("Ck", 0.0)

        Cp_factor = 2 if length_f in [1, 2] else 1
        C_trace = C_total - Cp_factor * (Cg + Ck)

        if C_trace <= 0:
            raise ValueError(
                "C_total must exceed the coupling capacitance contribution "
                f"(Cp_factor * (Cg + Ck) = {Cp_factor * (Cg + Ck):.3e} F); "
                "check that C_total is the full resonator self-capacitance."
            )

        length = C_trace / kwargs["wg"].C_m
        return cls.from_length_exact(length, **kwargs)

    @classmethod
    def coupling_strength_parameter(cls, Ck: float, Cg: float, frequency: float, R_L: float = 50.0):
        """Calculate the coupling strength parameter ωC_coupling*R_L."""
        C_coupling = Ck + Cg
        omega = 2 * np.pi * frequency
        return omega * C_coupling * R_L
    
    @classmethod
    def quarter_wave(cls, frequency: float, **kwargs):
        """Create a quarter-wavelength resonator at the specified frequency."""
        kwargs['length_f'] = 4
        return cls(frequency=frequency, **kwargs)
    
    @classmethod
    def half_wave(cls, frequency: float, **kwargs):
        """Create a half-wavelength resonator at the specified frequency."""
        kwargs['length_f'] = 2
        return cls(frequency=frequency, **kwargs)

    @classmethod
    def from_energies(cls, E_c: float, E_l: float, **kwargs):
        r"""
        Create a resonator explicitly defined by characteristic energies E_C and E_L.

        The physical length is solved using f0 = \sqrt{8\,E_C\,E_L} to preserve
        methods requiring a spatial dimension. The effective L and C properties
        are then forcibly overridden to exactly match the requested E_c and E_l,
        which is necessary for downstream capacitive coupling strength calculations.
        """
        from qfoundry.utils import E_to_C, E_to_L

        kwargs.setdefault("wg", cpw(11.45, 550, 15, 7.5, 0.2, material=kwargs.get("material")))
        frequency = np.sqrt(8.0 * E_c * E_l)
        resonator = cls(frequency=frequency, **kwargs)

        # Explicitly override the geometric derivations to enforce the exact
        # E_c / E_l split requested by the user.
        resonator._C_ = E_to_C(E_c)
        resonator._L_ = E_to_L(E_l)
        return resonator

    @classmethod
    def from_frequency(cls, frequency: float, **kwargs):
        """Create a resonator directly from a target resonance frequency."""
        kwargs.setdefault("wg", cpw(11.45, 550, 15, 7.5, 0.2, material=kwargs.get("material")))
        return cls(frequency=frequency, **kwargs)

    @classmethod
    def design_for_coupling(cls, frequency: float, Q_ext_target: float, **kwargs):
        """Design a resonator with a specific external Q factor by solving for Ck."""
        temp_resonator = cls(frequency=frequency, **kwargs)
        omega = 2 * np.pi * frequency
        C_k_required = np.sqrt(np.pi / (4 * Q_ext_target)) / (temp_resonator.wg.Z_0 * omega)

        kwargs['Ck'] = C_k_required
        return cls(frequency=frequency, **kwargs)

    @classmethod
    def design_for_kappa_ext(cls, frequency: float, kappa_ext_target_hz: float, Z_L: float = None,
                             x_ratio: float = None, **kwargs):
        """Design a resonator with a specific external coupling rate (kappa_ext,
        Hz) by solving for Ck. Companion to design_for_coupling() (which targets
        Q_ext — a different convention: wg.Z_0, no explicit Z_L).

        x_ratio: optional fractional coupler position x/L along the resonator.
        When given, the required Ck is scaled up by 1/voltage_ratio(x_ratio)
        so that the *effective* rate at that position hits the target."""
        from qfoundry.utils import kappa_ext_to_Ck
        kwargs.pop("Ck", None)
        temp_resonator = cls(frequency=frequency, **kwargs)
        if Z_L is None:
            Z_L = temp_resonator.wg.Z_0k or 50.0
        Ck = kappa_ext_to_Ck(frequency, kappa_ext_target_hz, temp_resonator.C(), Z_L)
        if x_ratio is not None:
            Ck /= temp_resonator.voltage_ratio(x_ratio)
        kwargs['Ck'] = Ck
        return cls(frequency=frequency, **kwargs)

    def _get_length_(self, f0, Cp: float = 0.0, n: int = 1):
        """
        Solves ω_n^2 * L(l) * C(l) = 1 for the physical length l.
        Matches the exact lumped L and C formulations used in __init__ to 
        guarantee consistency across all geometry types.
        """
        wg = self.wg

        def solve_quad(a, b, c):
            discriminant = np.sqrt(b**2 - 4 * a * c)
            return (-b + discriminant) / (2 * a), (-b - discriminant) / (2 * a)

        C_l = wg.C_m
        L_l = wg.L
        wn = 2 * np.pi * f0 * n 
        
        if self.length_f in [1, 2]:
            Ls = 2 * L_l / (self.n * np.pi) ** 2
        else:
            Ls = 8 * L_l / ((2 * self.n - 1) * np.pi) ** 2
            
        l1, l2 = solve_quad(C_l/2 * Ls * wn**2, Ls * Cp * wn**2, -1)
        return max(l1, l2) * self.length_f

    def Z_TL(self, f: np.array):
        fn = self.w0() / (2 * np.pi)
        Z = self.wg.Z_0k / (self.wg.alpha * self.length + 1j * np.pi * (f - fn) / fn)
        return Z / Z.max()

    def Zp(self, f):
        return self._Zp_(f, self.length_f)

    def Z(self, f):
        """Frequency domain numeric transfer function (impedance)."""
        return self.Z_TL(f) 

    def w0(self):
        return 2 * np.pi * self.f0()

    def f0(self):
        return self._f0_() 

    def f01(self):
        return self._f0_()
    
    def kappa(self):
        return self.f0() / self.Q()

    def voltage_ratio(self, x_ratio: float) -> float:
        """Normalised voltage amplitude |V(x)/V_max| at fractional position
        x_ratio = x/L along the resonator.

        For a quarter-wave resonator (length_f=4) x=0 is the shorted end and
        x=L the open end, so V(x) ∝ sin((2n-1)·πx/2L). Half- and full-wave
        resonators (length_f=2, 1) are open at both ends with V(x) a cosine,
        antinodes at the ends. A coupler placed at x_ratio sees its coupling
        rate scaled by voltage_ratio²  (Pozar/ Göppl et al. convention).
        """
        if not 0.0 <= x_ratio <= 1.0:
            raise ValueError(f"x_ratio must be within [0, 1], got {x_ratio}")
        if self.length_f == 4:
            return abs(np.sin((2 * self.n - 1) * np.pi * x_ratio / 2))
        # open-open resonators: full wavelength fits for length_f=1
        periods = self.n * (2 if self.length_f == 1 else 1)
        return abs(np.cos(periods * np.pi * x_ratio))

    def kappa_ext(self, Cin=None, Z_L: float = None, x_ratio: float = None):
        """External coupling rate (FWHM) due to coupling capacitance.

        x_ratio: optional fractional position x/L of the coupler along the
        resonator. When given, the rate is scaled by voltage_ratio(x_ratio)²
        — e.g. sin²(πx/2L) for a fundamental quarter-wave resonator. When
        None the coupler is assumed at a voltage antinode (maximum rate).
        """
        from qfoundry.utils import Ck_to_kappa_ext
        if Z_L is None:
            Z_L = self.wg.Z_0k or 50.0
        if Cin is None:
            Cin = self.Ck

        kappa = Ck_to_kappa_ext(self.f0(), Cin, self.C(), Z_L)
        if x_ratio is not None:
            kappa *= self.voltage_ratio(x_ratio) ** 2
        return kappa

    def Q_ext(self, Cin=None):
        """External quality factor due to coupling capacitance."""
        if Cin is None:
            Cin = self.Ck
        return np.pi / (4 * (self.wg.Z_0 * 2 * np.pi * self.f0() * Cin) ** 2)

    def Q_int(self):
        """Internal quality factor due to material losses.

        Q_int = R/(w0*L), the standard parallel-RLC result (equivalently
        w0*R*C, since w0*L = 1/(w0*C) at resonance) — matches the inherited
        circuit.Q() = R*sqrt(C/L).
        """
        return self._R_ / (self.w0() * self.L())
    
    def Q_total(self, Cin=None):
        """Loaded quality factor combining internal and external Q."""
        return 1 / (1/self.Q_int() + 1/self.Q_ext(Cin))
    
    def coupling_strength(self, Cin=None):
        """Ratio of internal to external Q (g = Q_int/Q_ext)."""
        return self.Q_int() / self.Q_ext(Cin)
    
    def transmission_coefficient(self, f, Cin=None):
        """Transmission coefficient |S₂₁|² for a side-coupled resonator."""
        if Cin is None:
            Cin = self.Ck
        g = self.coupling_strength(Cin)
        Q_tot = self.Q_total(Cin)
        df_over_f0 = (f - self.f0()) / self.f0()
        
        return g**2 / ((1 + g)**2 + (2 * Q_tot * df_over_f0)**2)
    
    def photon_number(self, power_dBm):
        """Average photon number in the resonator for a given input power."""
        power_watts = 10**(power_dBm/10 - 3) 
        hbar_omega = hbar * self.w0()
        return power_watts * self.Q_ext() / (hbar_omega * self.kappa_ext())
    
    def electric_field_rms(self, photon_number=1):
        """RMS electric field in the resonator for a given photon number."""
        A_eff = (self.wg.w + 2*self.wg.s) * self.wg.h 
        V_eff = A_eff * self.length * self.length_f 
        
        energy_per_photon = hbar * self.w0()
        energy_density = photon_number * energy_per_photon / V_eff
        
        return np.sqrt(2 * energy_density / (epsilon_0 * self.wg.epsilon_e))
    
    def participation_ratio(self, junction_area, gap_distance):
        """Calculate the participation ratio for a Josephson junction in the gap."""
        E_junction = self.electric_field_rms() * 2 
        U_junction = 0.5 * epsilon_0 * self.wg.epsilon_e * E_junction**2 * junction_area * gap_distance
        U_total = 0.5 * self.C() * (self.electric_field_rms() * gap_distance)**2
        
        return U_junction / U_total
    
    def dispersive_shift(self, alpha_qubit, participation_ratio):
        """Calculate the dispersive shift χ using the participation ratio."""
        E_c = e**2 / (2 * self.C()) 
        return alpha_qubit * participation_ratio * (E_c / (hbar * self.w0()))
    
    def purcell_rate(self, qubit_frequency, coupling_strength):
        """Calculate the Purcell decay rate for a coupled qubit."""
        detuning = abs(qubit_frequency - self.f0())
        if detuning == 0:
            raise ValueError("Qubit and resonator cannot be exactly on resonance")

        return coupling_strength**2 * self.kappa() / detuning**2
    
    def V_zpf(self):
        r"""Zero-point voltage fluctuation of the resonator."""
        return np.sqrt(hbar * self.w0() / (2 * self.C()))
    
    def V_rms(self, photon_number=0):
        """RMS voltage scaling for a given photon number."""
        quantum_scaling = np.sqrt(2 * photon_number + 1)
        return self.V_zpf() * quantum_scaling

    def critical_photon_number(self, critical_current):
        """Estimate the photon number where a connected junction becomes nonlinear."""
        I_rms_one_photon = self.w0() * self.C() * self.V_rms(photon_number=1)
        return (critical_current / (np.sqrt(2) * I_rms_one_photon))**2
    
    def fwhm(self, Cin=None):
        if Cin is None:
            Cin = self.Ck
        return self.f0() / self.Q_ext(Cin=Cin)

    def C(self):
        """Return the effective lumped capacitance of the resonator mode."""
        return self._C_
        
    def C_physical(self):
        """Return the total physical distributed capacitance of the CPW trace."""
        return self.wg.C_m * self.length

    def L(self):
        """Return the effective lumped inductance of the resonator mode."""
        return self._L_
    
    def __str__(self):
        """String representation with key resonator parameters."""
        return (
            "CPW Resonator Parameters:\n"
            "f₀ = \t%3.2f GHz \n"
            "L = \t%3.2f nH \n" 
            "C = \t%3.2f fF \n"
            "Q_int = \t%3.2f \n"
            "Q_ext = \t%3.2f \n"
            "κ_int = \t%3.2f MHz \n"
            "κ_ext = \t%3.2f MHz \n"
            "Length = \t%3.2f mm"
            % (
                self.f0() * 1e-9,
                self.L() * 1e9,
                self.C() * 1e15,
                self.Q_int(),
                self.Q_ext(),
                (self.f0() / self.Q_int()) * 1e-6,
                self.kappa_ext() * 1e-6,
                self.length * 1e3,
            )
        )