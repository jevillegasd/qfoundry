"""Waveguide utilities.
Includes coplanar waveguide (CPW) transmission line calculations.

"""
import numpy as np
from scipy import special as sp
from scipy.constants import c, epsilon_0

from qfoundry.materials import sc_metal, mu_0

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
        material: sc_metal, default=sc_metal()
            Superconducting metal of the CPW trace. Supplies the normal-state
            resistivity, critical temperature and London penetration depth
            used for the kinetic inductance and coherence length.
        alpha: float = 2.4e-2
            Attenuation coefficient in m⁻¹
        cm_x = 0.0e-12
            Capacitance correction per unit length in F/m

    Methods
    -------
    capacitances(w, s, h, eps_r)
        Calculate CPW capacitance per unit length and effective permittivity
        using Ghione (1984) elliptic integral formulation
    inductances(w, s, d, h)
        Calculate magnetic and kinetic inductances per unit length
        Kinetic inductance follows Wallraff et al. (2008) Eq. (A1)

    Notes
    -----
    The kinetic inductance calculation assumes the dirty limit approximation
    and uses the material's London penetration depth at its operating
    temperature; see qfoundry.materials.sc_metal.
    """

    def __init__(
        self,
        epsilon_r: float,  # Dielectric constant of the substrate
        height: float,  # [length], substrate's height in m
        width: float,  # [length], microstrip width in m
        spacing: float,  # [length], Space from ground plane in m
        thickness: float,
        material: sc_metal = None,  # superconducting metal of the CPW trace
        alpha: float = 2.4e-2,  # attenuation cofficient m^-1
        cm_x=0.0e-12,  # capacitance corection per unit length in F/m
    ):

        # if dimensional units are large, assume they are in um
        if width > 1e-3 or thickness > 1e-3 or spacing > 1e-3:
            width = width * 1e-6
            height = height * 1e-6
            spacing = spacing * 1e-6
            thickness = thickness * 1e-6

        self.w = width  # to match [1]
        self.s = spacing  # to match [1]
        self.d = thickness  # to match [1]
        self.h = height

        self.material = material if material is not None else sc_metal()
        self.rho_tc = self.material.rho
        self.tc = self.material.Tc
        self.Lambda_L = self.material.london_penetration_depth()
        self.lambda_0 = self.material.coherence_length()

        self.alpha = alpha  # attenuation cofficient in m^-1

        self.L_m, self.L_k = self.inductances(self.w, self.s, self.d, self.h)
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
            mu_0 * (self.Lambda_L)**2  / ((d * w)) * g
        )  # Thin-film (d << Lambda_L) kinetic inductance, Watanabe et al., Jpn. J. Appl. Phys. 33, 5708 (1994), doi:10.1143/JJAP.33.5708

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
