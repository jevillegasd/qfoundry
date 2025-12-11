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

from scipy.constants import c, epsilon_0, m_e, hbar, e, k, pi, Avogadro
import numpy as np
from scipy import special as sp
import scqubits as scq


Avogadro = 6.022e23  # atoms per mol
Al_mass = 26.98e-3  # kg/mol
Al_density = 2.7e3  # kg/m^3
n_Al = Avogadro * Al_density / Al_mass  # atoms / m^3
mu_0 = 4 * np.pi * 1e-7  # H/m


class cpw:
    """
    A coplanar waveguide transmission line.
    
    Implements the CPW impedance and effective permittivity calculations from
    Ghione (1984) using elliptic integrals. The kinetic inductance is calculated
    following the model in Wallraff et al. (2008).

    Attributes
    ----------
        epsilon_r: float
            Dielectric constant of the substrate
        height: float
            Substrate height in m (converted from μm if > 1e-3)
        width: float
            Center conductor width in m (converted from μm if > 1e-3)
        spacing: float
            Gap width from center conductor to ground in m (converted from μm if > 1e-3)
        thickness: float = 100e-9
            Superconducting metal layer thickness in m
        rho: float = 2.06e-9
            Normal state resistivity of the thin film in Ω⋅m
        tc: float = 1.23
            Critical temperature in K
        alpha: float = 2.4e-2
            Attenuation coefficient in m⁻¹
        n_s = 3*n_Al
            Superconducting electron density in m⁻³
        T = 20e-3
            Operating temperature in K
        cm_x = 0.0e-12
            Capacitance correction per unit length in F/m

    Methods
    -------
    capacitances(w, s, h, eps_r)
        Calculate CPW capacitance per unit length and effective permittivity
        using Ghione (1984) elliptic integral formulation
    inductances(w, s, d, h, rho, tc)
        Calculate magnetic and kinetic inductances per unit length
        Kinetic inductance follows Wallraff et al. (2008) Eq. (A1)
    
    Notes
    -----
    The kinetic inductance calculation assumes the dirty limit approximation
    and uses the London penetration depth with temperature dependence.
    """

    def __init__(
        self,
        epsilon_r: float,  # Dielectric constant of the substrate
        height: float,  # [length], substrate's height in m
        width: float,  # [length], microstrip width in m
        spacing: float,  # [length], Space from ground plane in m
        thickness: float = 100e-9,
        rho: float = 2.06e-9,  # normal state resisitivity of the thin film
        tc: float = 1.23,  # critical temperature in K
        alpha: float = 2.4e-2,  # attenuation cofficient m^-1
        n_s=3 * n_Al,  # superconducting electron density in m^-3
        T=20e-3,
        cm_x=0.0e-12,  # capacitance corection per unit length in F/m
    ):  # temperature in K

        # if dimensional units are large, assume they are in um
        if width > 1e-3 or height > 1e-3 or spacing > 1e-3:
            width = width * 1e-6
            height = height * 1e-6
            spacing = spacing * 1e-6
            thickness = thickness * 1e-6

        self.w = width  # to match [1]
        self.s = spacing  # to match [1]
        self.d = thickness  # to match [1]
        self.h = height
        self.rho_tc = rho
        self.tc = tc

        # London penetration length alternative
        Lambda_0 = np.sqrt(m_e / (mu_0 * n_s * e**2))  # London penetration length in m
        self.Lambda_L = Lambda_0 * (1 - (T / tc) ** 4) ** (
            -0.5
        )  # Effective London penetration length in m (https://rashid-phy.github.io/me/pdf/notes/Superconductor_Theory.pdf eq. 24)

        self.alpha = alpha  # attenuation cofficient in m^-1
        self.lambda_0 = 1.05e-3 * np.sqrt(self.rho_tc / self.tc)  # Cohenrece Length

        self.L_m, self.L_k = self.inductances(self.w, self.s, self.d, self.h, rho, tc)
        self.C_m, self.epsilon_e = self.capacitances(self.w, self.s, self.h, epsilon_r)
        self.C_m += cm_x
        self.L = self.L_m + self.L_k

        self.Z_0 = np.sqrt(self.L_m / self.C_m)  # Equation (1) in [1]
        self.Z_0k = np.sqrt(self.L / self.C_m)
        self.epsilon_ek = c**2 * (self.C_m * self.L)
        self.eta_0 = mu_0 * c  # ~120*np.pi

    def LCR_f(L, C, R) -> float:
        return 1 / np.sqrt(L * C) / (2 * np.pi)

    def capacitances(self, w: float, s: float, h: float, eps_r: float):
        """
        Calculate capacitances and effective permittivity of CPW
        """
        k_0 = w / (w + 2 * s)
        k_1 = np.sinh(np.pi * w / (4 * h)) / np.sinh((np.pi * (w + 2 * s)) / (4 * h))
        k_0p = np.sqrt(1 - np.square(k_0))  # k'_0 in book notation
        k_11 = np.sqrt(1 - np.square(k_1))  # k'_1 in book notation

        K0 = sp.ellipk(k_0)
        K0p = sp.ellipk(k_0p)
        K1 = sp.ellipk(k_1)
        K1p = sp.ellipk(k_11)

        # Equation (2) in [1]
        eps_eff = 1 + (eps_r - 1) * (K1 * K0p) / (2 * K1p * K0)
        C_m = 4 * epsilon_0 * eps_eff * K0 / K0p
        return C_m, eps_eff

    def inductances(
        self,
        w: float,
        s: float,
        d: float,
        h: float,
        rho: float = 2.06e-3,
        tc: float = 1.23,
    ):
        """
        Calculate normal and kinetic inductances of a CPW
        """
        k_0 = w / (w + 2 * s)
        k_1 = np.sinh(np.pi * w / (4 * h)) / np.sinh((np.pi * (w + 2 * s)) / (4 * h))
        k_0p = np.sqrt(1 - np.square(k_0))  # k'_0 in book notation
        k_11 = np.sqrt(1 - np.square(k_1))  # k'_1 in book notation

        K0 = sp.ellipk(k_0)
        K0p = sp.ellipk(k_0p)
        K1 = sp.ellipk(k_1)
        K1p = sp.ellipk(k_11)

        ######## Inductance per unit length #####################
        L_m = mu_0 / 4 * K0p / K0

        ######### Kinetic inductance ############################
        A = 1 / (2 * k_0**2 * K0**2)
        g = A * (
            -np.log(d / (4 * w))
            - k_0 * np.log(d / (4 * (w + 2 * s)))
            + (2 * (w + s) / (w + 2 * s)) * np.log(s / (w + s))
        )

        L_k = (
            mu_0 * (self.Lambda_L**2) / ((d * w))
        )  # Equation (2) in [1], Qualitatively, in the limit S << W [...] the kinetic contribution reduces to Lk = mu_0*Lambda_L^2/W

        # [1]  10.1063/1.4962172
        return L_m, L_k
    
    def phase_velocity(self):
        """
        Phase velocity of the CPW mode in m/s.
        
        Returns
        -------
        float
            Phase velocity v_p = c/√ε_eff
        """
        return c / np.sqrt(self.epsilon_e)
    
    def wavelength(self, frequency):
        """
        Wavelength in the CPW at given frequency.
        
        Parameters
        ----------
        frequency : float
            Frequency in Hz
            
        Returns
        -------
        float
            Wavelength λ = v_p/f in m
        """
        return self.phase_velocity() / frequency
    
    def characteristic_impedance_kinetic(self):
        """
        Characteristic impedance including kinetic inductance.
        
        Returns
        -------
        float
            Z₀ = √((L_m + L_k)/C_m) in Ohms
        """
        return self.Z_0k
    
    def loss_tangent(self, frequency):
        """
        Loss tangent of the CPW at given frequency.
        
        Parameters
        ----------
        frequency : float
            Frequency in Hz
            
        Returns
        -------
        float
            tan(δ) = R/(ωL) where R is the series resistance per unit length
        """
        omega = 2 * np.pi * frequency
        R_per_length = self.alpha * self.Z_0k  # Series resistance per unit length
        return R_per_length / (omega * self.L)
    
    def __str__(self):
        return f"Coplanar Waveguide: {self.L_m*1e6:3.2f} uH/m (magnetic), {self.L_k*1e6:3.2f} uH/m (kinetic), {self.C_m*1e12:3.2f} pF/m, Z0 = {self.Z_0:3.2f} Ohm, epsilon_eff = {self.epsilon_e:3.2f}"

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
        w = 2 * np.pi * self._f0_()
        return self.R() + +1j * w * self.L() + 1 / (1j * w * self.C())

    def _Zp_(self, f, n):
        w = 2 * np.pi * self._f0_()
        return 1 / (1 / self.R() + 1 / (1j * w * self.L() * n) + 1j * w * self.C() * n)

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
        return self.Z + o.Z

    def __multiply__(self, o):
        return self.Z * o.Z

    def R(self):
        return self._R_

    def C(self):
        return self._C_

    def L(self):
        return self._L_


class cpw_resonator(circuit):
    """
    A coplanar waveguide resonator based on Wallraff et al. (2008).
    
    Implements the distributed LC model where the resonator is treated as a
    lumped circuit with effective L and C values derived from the CPW geometry.
    
    The resonator frequency is given by f₀ = 1/(2π√LC) where:
    - L = 2*L_l*l/(n*π)² (Wallraff Eq. 11)
    - C = C_l*l/2 + C_c (Wallraff Eq. 12)
    
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
        Effective coupling capacitance accounting for load impedance (Wallraff Eq. 15)
    qmodel : scqubits.Oscillator
        Quantum model of the resonator for multi-level calculations
        
    Methods
    -------
    f0()
        Resonance frequency in Hz
    Q_ext(Cin=None)
        External quality factor due to coupling
    kappa_ext()
        External coupling rate (FWHM) in Hz
    Z_TL(f)
        Transmission line impedance as function of frequency
        
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
        R_L: float = 50.0,  # Load resistance in Ohms
        **kwargs
    ):
        self.wg = wg
        
        if frequency is None:
            raise ValueError("Frequency must be provided, or object instance must be created with a length using from_length method.")

        if int(length_f) not in [1, 2, 4]:
            raise ValueError("length_f must be 1 (full-wave), 2 (half-wave) or 4 (quarter-wave)")
        self.length_f = length_f  # length factor: 4: quarter wavelength resonator (short in one end)
        self.n = n  # mode number
        self.Cin = Ck

        wn = 2 * np.pi * frequency * n  # Angular frequency of the resonator mode
        
        if length_f in [1, 2]:
            eff_k = 1 # Full coupling at antinode
            Cp_factor = 2
        else:  # Quarter-wave resonator
            eff_k = 0  # Negligible contribution of shorted end to coupling capacitance
            Cp_factor = 2 # Should be 1 for single count of antinode capacitance??
        C_k = eff_k*Ck + Cg

        Cp = C_k / (1 + wn**2 * C_k ** 2 * R_L**2) # https://arxiv.org/pdf/0807.4094 (Wallraff2008) [15]
        
        self.Cp = Cp # Effective coupling capacitance including load impedance effect
        if self.length is None:
            self.length = self._get_length_(
                frequency, Cp*Cp_factor , n = n
            )/ self.length_f

        # Wallraff et al. (2008) Eq. (11): L = 2*L_l*l/(n*π)²
        self._L_ = (
            2 * self.wg.L * self.length / (self.n * 2 * np.pi) ** 2  # The factor of 1/2**2 cancels out when length_f=2, like in Wallraff Eq. 11
        ) * self.length_f

        # Wallraff et al. (2008) Eq. (12): C = C_l*l/2 + C_c
        self._C_ = (self.wg.C_m /2 * self.length* self.length_f  + self.Cp*Cp_factor)

        self._R_ = wg.Z_0k / (self.wg.alpha * self.length )
         # Wallraff et al. (2008) Eq. (13): R = Z0/(alpha*l

        # Correct for the coupling capacitance 
        self._R_ += (1 + wn**2 * (Ck + Cg) ** 2 * R_L**2) / (
            wn**2 * (Ck + Cg + 1e-20) ** 2 * R_L
        )
       

        if self.qmodel is None and kwargs.get("inst_model", True):
            self.truncated_dim = kwargs.get("truncated_dim", 4)
            self.qmodel = scq.Oscillator(
                E_osc=self.f0() * 1e-9,
                l_osc=self.length,
                truncated_dim=self.truncated_dim,  # up to 3 photons (0,1,2,3)
            )
            
    
    @classmethod
    def from_length(cls, length: float, **kwargs):
        """
        Create a resonator from a given length.
        
        Uses the distributed LC model from Wallraff et al. (2008) Eq. (11-12).
        Iteratively solves for the correct frequency when coupling capacitances
        are large, accounting for the frequency dependence of Cp.
        
        Parameters
        ----------
        length : float
            Physical length of the resonator in m
        **kwargs
            Additional parameters passed to __init__
        """
        
        wg = kwargs.get("wg", cpw(11.7, 0.1, 15, 7.5, alpha=2.4e-2))
        length_f = kwargs.get("length_f", 2) 
        n = kwargs.get("n", 1)
        Ck = kwargs.get("Ck", 0.0) # Coupling capacitance to feedline in F
        Cg = kwargs.get("Cg", 0.0) # Coupling capacitance to at antinode in F
        R_L = kwargs.get("R_L", 50.0)
        
        # Total nominal coupling capacitance
         # Coupling efficiency factor, 1 for full coupling, <1 for partial eff_k = V(x)^2/ max|V(x)|^2
        
        if length_f in [1, 2]:
            eff_k = 1 # Full coupling at antinode
            Cp_factor = 2
        else:  # Quarter-wave resonator
            eff_k = 0  # Negligible contribution of shorted end to coupling capacitance
            Cp_factor = 2 # Should be 1 for single count of antinode capacitance??

        C_k = eff_k*Ck + Cg
        # For large coupling capacitances, iteratively solve for frequency
        # Start with weak coupling approximation
        
        L = (2 * wg.L * length / (n * 2 * np.pi) ** 2) * length_f
        C = (wg.C_m * length / 2 * length_f + C_k*Cp_factor)
        f0_guess = 1 / (2 * np.pi * np.sqrt(
            L * C
        ))

        # For small coupling capacitances, use simple approximation
        if C_k < 1e-15:  # Less than 1 fF
            print("Using weak coupling approximation for resonator frequency.")
            return cls(frequency=f0_guess, **kwargs)
        
        # Iterative solution for frequency-dependent Cp
        max_iterations = 50
        tolerance = 1e-7  # Relative frequency tolerance
        
        for iteration in range(max_iterations):
            wn = 2 * np.pi * f0_guess
            
            # Calculate frequency-dependent Cp (Wallraff Eq. 15)
            Cp = C_k / (1 + wn**2 * C_k**2 * R_L**2)
            
            # Calculate new frequency with corrected Cp
            C = (wg.C_m * length * length_f)  / 2 + Cp * Cp_factor
            L = (2 * wg.L * length / (n * 2 * np.pi) ** 2) * length_f
            f0_new = 1 / (2 * np.pi * np.sqrt(L * C)) 
            
            # Check for convergence
            relative_error = abs(f0_new - f0_guess) / f0_guess
            if relative_error < tolerance:
                break
            f0_guess = f0_new
            
        if iteration == max_iterations - 1:
            import warnings
            warnings.warn(f"Frequency iteration did not converge after {max_iterations} iterations. "
                         f"Final relative error: {relative_error:.2e}")
        
        # Create instance and set length as instance attribute
        instance = cls(frequency=f0_new, **kwargs)
        instance.length = length
        return instance
    
    @classmethod
    def from_length_exact(cls, length: float, **kwargs):
        """
        Create a resonator from a given length using exact root finding.
        
        Solves the implicit equation for frequency accounting for the full
        frequency dependence of Cp using scipy.optimize.fsolve.
        This is more robust than the iterative method for strongly coupled cases.
        
        Parameters
        ----------
        length : float
            Physical length of the resonator in m
        **kwargs
            Additional parameters passed to __init__
        """
        from scipy.optimize import fsolve
        
        wg = kwargs.get("wg", cpw(11.7, 0.1, 12, 6, alpha=2.4e-2))
        length_f = kwargs.get("length_f", 2) 
        n = kwargs.get("n", 1)
        Ck = kwargs.get("Ck", 0.0)
        Cg = kwargs.get("Cg", 0.0)
        R_L = kwargs.get("R_L", 50.0)
        
        C_coupling = Ck + Cg
        
        # For small coupling capacitances, use simple calculation
        if C_coupling < 1e-15:  # Less than 1 fF
            Cp = C_coupling
            C = wg.C_m * length * length_f / 2 + Cp
            L = 2 * wg.L * length * length_f / (n * np.pi) ** 2
            f0 = 1 / (2 * np.pi * np.sqrt(L * C))
            return cls(frequency=f0, **kwargs)
        
        def frequency_equation(f0):
            """
            Implicit equation: f0 = 1/(2π√LC) where C depends on f0 through Cp.
            Returns zero when the equation is satisfied.
            """
            wn = 2 * np.pi * f0 * n
            Cp = C_coupling / (1 + wn**2 * C_coupling**2 * R_L**2)
            
            C = wg.C_m * length * length_f / 2 + 2 * Cp
            L = 2 * wg.L * length * length_f / (n * np.pi) ** 2
            
            f0_calculated = 1 / (2 * np.pi * np.sqrt(L * C))
            return f0 - f0_calculated
        
        # Initial guess using weak coupling approximation
        f0_guess = 1 / (2 * np.pi * np.sqrt(
            (2 * wg.L * length * length_f / (n * np.pi) ** 2) * 
            (wg.C_m * length * length_f / 2 + C_coupling)
        ))
        
        # Solve the implicit equation
        try:
            f0_solution = fsolve(frequency_equation, f0_guess, xtol=1e-12)[0]
        except Exception as e:
            import warnings
            warnings.warn(f"Root finding failed: {e}. Using iterative method as fallback.")
            return cls.from_length(length, **kwargs)
        
        # Create instance and set length as instance attribute
        instance = cls(frequency=f0_solution, **kwargs)
        instance.length = length
        return instance
        
    @classmethod
    def coupling_strength_parameter(cls, Ck: float, Cg: float, frequency: float, R_L: float = 50.0):
        """
        Calculate the coupling strength parameter ωC_coupling*R_L.
        
        This parameter determines whether the weak coupling approximation is valid:
        - << 1: Weak coupling, simple Cp ≈ Ck + Cg approximation is accurate
        - >> 1: Strong coupling, frequency-dependent Cp calculation needed
        
        Parameters
        ----------
        Ck : float
            Coupling capacitance to feedline in F
        Cg : float
            Coupling capacitance to ground in F  
        frequency : float
            Frequency in Hz
        R_L : float, default=50.0
            Load resistance in Ohms
            
        Returns
        -------
        float
            Coupling strength parameter ωC_total*R_L
        """
        C_coupling = Ck + Cg
        omega = 2 * np.pi * frequency
        return omega * C_coupling * R_L
    
    @classmethod
    def quarter_wave(cls, frequency: float, **kwargs):
        """
        Create a quarter-wavelength resonator at the specified frequency.
        
        Parameters
        ----------
        frequency : float
            Target resonance frequency in Hz
        **kwargs
            Additional parameters passed to __init__
        """
        kwargs['length_f'] = 4
        return cls(frequency=frequency, **kwargs)
    
    @classmethod  
    def half_wave(cls, frequency: float, **kwargs):
        """
        Create a half-wavelength resonator at the specified frequency.
        
        Parameters
        ----------
        frequency : float
            Target resonance frequency in Hz
        **kwargs
            Additional parameters passed to __init__
        """
        kwargs['length_f'] = 2
        return cls(frequency=frequency, **kwargs)
    
    @classmethod
    def design_for_coupling(cls, frequency: float, Q_ext_target: float, **kwargs):
        """
        Design a resonator with specified external Q factor.
        
        Parameters
        ----------
        frequency : float
            Target resonance frequency in Hz
        Q_ext_target : float
            Target external quality factor
        **kwargs
            Additional parameters passed to __init__
            
        Returns
        -------
        cpw_resonator
            Resonator with coupling capacitance set to achieve target Q_ext
        """
        # Create initial resonator to get Z_0
        temp_resonator = cls(frequency=frequency, **kwargs)
        
        # Calculate required coupling capacitance for target Q_ext
        # From Q_ext = π/(4*(Z₀*ω*C_k)²)
        omega = 2 * np.pi * frequency
        C_k_required = np.sqrt(np.pi / (4 * Q_ext_target)) / (temp_resonator.wg.Z_0 * omega)
        
        kwargs['Ck'] = C_k_required
        return cls(frequency=frequency, **kwargs)

    def _get_length_(self, f0, Cp: float = 0.0, n: int = 1):
        from scipy.constants import c as c0
        # Solve quadratic equation for length when Cp is significant
        wg = self.wg

        def solve_quad(a, b, c):
            return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a), (
                -b - np.sqrt(b**2 - 4 * a * c)
            ) / (2 * a)

        # If Cg + Ck == 0, the length is calculated using only the cpw
        if Cp > 1e-20:
            C_l = wg.C_m
            L_l = wg.L
            wn = 2 * np.pi * f0 * n # Angular frequency of the resonator mode
            Ls = 2 * L_l / (2 * self.n * np.pi) ** 2
            l1, l2 = solve_quad(C_l/2 * Ls * wn**2, Ls * Cp * wn**2, -1)
            return max(l1, l2)
        else:
            return ((c0) / (f0 * n * (wg.epsilon_ek**0.5)))

    def Z_TL(self, f: np.array):
        fn = self.w0() / (2 * np.pi)
        Z = self.wg.Z_0k / (self.wg.alpha * self.length + 1j * np.pi * (f - fn) / fn)
        return Z / Z.max()

    def Zp(self, f):
        return self._Zp_(f, self.length_f)

    def Z(self, f):
        """
        frequency domain numeric transfer function (impedance), this overload the class:circuits impedance.
        """
        return self.Z_TL(f)  # Overload this function from the circuit class

    def w0(self):
        return 2 * np.pi * self.f0()

    def f0(self):
        return (
            self._f0_()
        )  # __f0__() calculates the fundamental LC resonance of the RCL circuit

    def f01(self):
        return self._f0_()
    
    def kappa(self):
        return self.f0() / self.Q()

    def kappa_ext(self, Cin=None, Z_L: float = 50.0):
        """
        External coupling rate (FWHM) due to coupling capacitance.
        κ_ext = (ω₀*C_k)²*Z_L / C
        """
        if Cin is None:
            Cin = self.Cin
        kappa_ = (self.w0()*self.Cin)**2*Z_L / (self.C())
        return kappa_ / (2 * np.pi)

    def Q_ext(self, Cin=None):
        """
        External quality factor due to coupling capacitance.
        
        Q_ext = π/(4*(Z₀*ω*C_k)²)
        """
        if Cin == None:
            Cin = self.Cin
        return np.pi / (4 * (self.wg.Z_0 * 2 * np.pi * self.f0() * Cin) ** 2)
        # return (1+(wr*C_k*R_L)**2)*(C+C_k))/(wr*C_k**2*R_L)  R_L=50 Ohm

    def Q_int(self):
        """
        Internal quality factor due to material losses.
        
        Q_int = ω₀*L/R_s where R_s is the series resistance
        """
        return self.w0() * self.L() / self._R_
    
    def Q_total(self, Cin=None):
        """
        Total quality factor: 1/Q_total = 1/Q_int + 1/Q_ext
        
        This is the loaded quality factor measured in experiments.
        """
        return 1 / (1/self.Q_int() + 1/self.Q_ext(Cin))
    
    def coupling_strength(self, Cin=None):
        """
        Coupling strength parameter g = Q_int/Q_ext.
        
        g << 1: undercoupled, g >> 1: overcoupled, g ≈ 1: critically coupled
        """
        return self.Q_int() / self.Q_ext(Cin)
    
    def transmission_coefficient(self, f, Cin=None):
        """
        Transmission coefficient |S₂₁|² for a side-coupled resonator.
        
        For a resonator side-coupled to a transmission line, the transmission
        depends on the coupling strength and frequency detuning.
        Standard form: |S₂₁|² = |1 - (κ_ext/2)/(iΔ + κ_total/2)|²
        where Δ = f - f₀ and κ_total = κ_int + κ_ext.
        """
        if Cin is None:
            Cin = self.Cin
        g = self.coupling_strength(Cin)
        Q_tot = self.Q_total(Cin)
        df_over_f0 = (f - self.f0()) / self.f0()
        
        return g**2 / ((1 + g)**2 + (2 * Q_tot * df_over_f0)**2)
    
    def photon_number(self, power_dBm):
        """
        Average photon number in the resonator for given input power.
        
        Parameters
        ----------
        power_dBm : float
            Input power in dBm
            
        Returns
        -------
        float
            Average photon number ⟨n⟩
            
        Notes
        -----
        For a driven resonator, the steady-state photon number depends on
        input power, coupling strength, and internal losses. The exact
        relationship depends on the driving configuration and impedance matching.
        ⟨n⟩ = P_in * Q_ext / (ℏω₀ * κ_ext) for on-resonance driving.
        """
        power_watts = 10**(power_dBm/10 - 3)  # Convert dBm to watts
        hbar_omega = hbar * self.w0()
        kappa_ext = self.kappa_ext()
        
        return power_watts * self.Q_ext() / (hbar_omega * kappa_ext)
    
    def electric_field_rms(self, photon_number=1):
        """
        RMS electric field in the resonator for given photon number.
        
        Parameters
        ----------
        photon_number : float, default=1
            Number of photons in the resonator
            
        Returns
        -------
        float
            RMS electric field in V/m
            
        Notes
        -----
        Uses the relation E_rms = √(ℏω₀/(2ε₀V_eff)) * √n where V_eff is the
        effective mode volume. For a CPW resonator, V_eff ≈ A_eff * length
        where A_eff is the effective cross-sectional area.
        """
        # Effective area for CPW (approximate)
        A_eff = (self.wg.w + 2*self.wg.s) * self.wg.h  # Cross-sectional area
        V_eff = A_eff * self.length * self.length_f  # Effective mode volume
        
        energy_per_photon = hbar * self.w0()
        energy_density = photon_number * energy_per_photon / V_eff
        
        # E_rms = √(2*U_E/(ε₀*ε_eff*V)) where U_E is electric energy
        return np.sqrt(2 * energy_density / (epsilon_0 * self.wg.epsilon_e))
    
    def participation_ratio(self, junction_area, gap_distance):
        """
        Calculate the participation ratio for a Josephson junction.
        
        Parameters
        ----------
        junction_area : float
            Area of the Josephson junction in m²
        gap_distance : float
            Gap distance of the junction in m
            
        Returns
        -------
        float
            Participation ratio p_j
            
        Notes
        -----
        The participation ratio quantifies how much of the resonator's electric
        field energy is concentrated in the junction. This is commonly defined as:
        p_j = U_j / U_total where U_j is the electric energy in the junction
        and U_total is the total electric energy stored in the resonator.
        
        See: Nigg et al. (2012) Phys. Rev. Lett. 108, 240502 for transmon theory
        and Krantz et al. (2019) Appl. Phys. Rev. 6, 021318 for review.
        """
        # Electric field in the junction (simplified as uniform field)
        E_junction = self.electric_field_rms() * 2  # Factor of 2 for field enhancement
        
        # Electric energy in the junction
        U_junction = 0.5 * epsilon_0 * self.wg.epsilon_e * E_junction**2 * junction_area * gap_distance
        
        # Total electric energy in resonator
        U_total = 0.5 * self.C() * (self.electric_field_rms() * gap_distance)**2
        
        return U_junction / U_total
    
    def dispersive_shift(self, alpha_qubit, participation_ratio):
        """
        Calculate the dispersive shift χ for qubit-resonator coupling.
        
        Parameters
        ----------
        alpha_qubit : float
            Qubit anharmonicity in Hz (negative for transmon)
        participation_ratio : float
            Participation ratio of the qubit junction
            
        Returns
        -------
        float
            Dispersive shift χ in Hz
            
        Notes
        -----
        In the dispersive regime, the shift is approximately:
        χ ≈ g²/Δ where g is the coupling strength and Δ is the detuning.
        For transmons, this can also be written in terms of participation ratio.
        
        See: Koch et al. (2007) Phys. Rev. A 76, 042319 for transmon theory
        and Blais et al. (2004) Phys. Rev. A 69, 062320 for dispersive coupling.
        """
        # Charging energy from participation ratio (simplified)
        E_c = e**2 / (2 * self.C())  # Charging energy in Joules
        
        return alpha_qubit * participation_ratio * (E_c / (hbar * self.w0()))
    
    def purcell_rate(self, qubit_frequency, coupling_strength):
        """
        Calculate the Purcell decay rate for a qubit coupled to the resonator.
        
        Parameters
        ----------
        qubit_frequency : float
            Qubit transition frequency in Hz
        coupling_strength : float
            Qubit-resonator coupling strength g in Hz
            
        Returns
        -------
        float
            Purcell decay rate in Hz
            
        Notes
        -----
        The Purcell effect enhances spontaneous emission into the resonator mode.
        Γ_Purcell = (g²/Δ²) * κ where g is coupling, Δ is detuning, κ is decay rate.
        
        See: Purcell (1946) Phys. Rev. 69, 37 for original theory
        and Houck et al. (2007) Nature 449, 328 for circuit QED implementation.
        """
        detuning = abs(qubit_frequency - self.f0())
        if detuning == 0:
            raise ValueError("Qubit and resonator cannot be exactly on resonance")
            
        return coupling_strength**2 * self.kappa_ext() / detuning**2
    
    def voltage_rms(self, photon_number=1):
        """
        RMS voltage across the resonator for given photon number.
        
        Parameters
        ----------
        photon_number : float, default=1
            Number of photons in the resonator
            
        Returns
        -------
        float
            RMS voltage in V
            
        Notes
        -----
        Uses V_rms = √(ℏω₀/(2C)) * √n for the voltage across the capacitor.
        This is useful for calculating nonlinear effects and power handling.
        """
        return np.sqrt(hbar * self.w0() * photon_number / (2 * self.C()))
    
    def critical_photon_number(self, critical_current):
        """
        Estimate the critical photon number for onset of nonlinearity.
        
        Parameters
        ----------
        critical_current : float
            Critical current of any Josephson junctions in the circuit in A
            
        Returns
        -------
        float
            Critical photon number n_crit
            
        Notes
        -----
        This estimates when the RF current approaches the critical current,
        marking the onset of strong nonlinearity. Uses I_RF = ω₀ * C * V_RMS.
        """
        # RF current for one photon
        I_rf_one_photon = self.w0() * self.C() * self.voltage_rms(photon_number=1)
        
        return (critical_current / I_rf_one_photon)**2

    def fwhm(self, Cin=None):
        if Cin == None:
            Cin = self.Cin
        return self.f0() / self.Q_ext(Cin=Cin)

    def C(self):
        """Return the total capacitance of the resonator."""
        return self._C_

    def L(self):
        """Return the total inductance of the resonator."""
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
        

# References with specific equation numbers and accurate citations:
#
# [1] Ghione (1984), doi: 10.1049/el:19840120
#     - Elliptic integral formulation for CPW impedance and capacitance
#
# [2] Watanabe (1994), doi: 10.1143/JJAP.33.5708  
#     - CPW effective permittivity calculations
#
# [3] Wallraff et al. (2008), arXiv:0807.4094
#     - Eq. (11): Resonator inductance L = 2*L_l*l/(n*π)²
#     - Eq. (12): Resonator capacitance C = C_l*l/2 + C_c  
#     - Eq. (15): Coupling capacitance C_p = C_k/(1 + (ω*C_k*R_L)²)
#     - General theory of CPW resonators and quality factors
#
# Additional key references for quantum circuit theory:
# - Koch et al. (2007) Phys. Rev. A 76, 042319 - Transmon qubit theory
# - Blais et al. (2004) Phys. Rev. A 69, 062320 - Circuit QED theory  
# - Nigg et al. (2012) Phys. Rev. Lett. 108, 240502 - Participation ratio
# - Krantz et al. (2019) Appl. Phys. Rev. 6, 021318 - Comprehensive review
# - Houck et al. (2007) Nature 449, 328 - Circuit QED experiments
# - Purcell (1946) Phys. Rev. 69, 37 - Original Purcell effect theory
#
