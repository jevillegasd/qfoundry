"""Superconducting material properties.

Models a thin-film superconductor from its critical temperature, normal-state
resistivity and superconducting carrier density. These properties feed the
BCS superconducting gap (used for Josephson junction Ic/Rn conversions) and
the London penetration depth / coherence length (used for CPW kinetic
inductance calculations).

References
----------
- Tinkham, Introduction to Superconductivity - BCS gap and coherence length
- Wallraff et al. (2008), arXiv:0807.4094 - CPW kinetic inductance
"""

import numpy as np
from scipy.constants import m_e, e
from scipy.constants import elementary_charge as e_0
from scipy.constants import Boltzmann as k_B

Avogadro = 6.022e23  # atoms per mol
Al_mass = 26.98e-3  # kg/mol
Al_density = 2.7e3  # kg/m^3
n_Al = Avogadro * Al_density / Al_mass  # atoms / m^3
mu_0 = 4 * np.pi * 1e-7  # H/m


class sc_metal:
    """
    Superconducting thin-film metal.

    Parameters
    ----------
    Tc : float, default=1.14
        Critical temperature in K.
    T : float, default=20e-3
        Operating temperature in K.
    rho : float, default=2.06e-9
        Normal-state resistivity of the thin film in Ohm*m.
    n_s : float, default=3*n_Al
        Superconducting electron density in m^-3.

    Notes
    -----
    Default values (Tc, rho, n_s) correspond to a thin aluminum film.
    """

    def __init__(
        self,
        Tc: float = 1.14,
        T: float = 20e-3,
        rho: float = 2.06e-9,
        n_s: float = 3 * n_Al,
    ):
        self.Tc = Tc
        self.T = T
        self.rho = rho
        self.n_s = n_s

    def sc_gap(self):
        """BCS superconducting gap Delta(T) in Joules."""
        if self.T < 0.1:
            return 1.764 * k_B * self.Tc
        else:
            return 3.076 * k_B * np.sqrt(1 - self.T / self.Tc)

    def sc_gap_eV(self):
        """BCS superconducting gap Delta(T) in eV."""
        return self.sc_gap() / e_0

    def london_penetration_depth_0(self):
        """London penetration depth at T=0, in m."""
        return np.sqrt(m_e / (mu_0 * self.n_s * e**2))

    def london_penetration_depth(self, T: float = None):
        """
        Effective (temperature-corrected) London penetration depth, in m.

        https://rashid-phy.github.io/me/pdf/notes/Superconductor_Theory.pdf eq. 24

        Parameters
        ----------
        T : float, optional
            Temperature in K. Defaults to the material's operating temperature (self.T).
        """
        if T is None:
            T = self.T
        return self.london_penetration_depth_0() * (1 - (T / self.Tc) ** 4) ** (-0.5)

    def coherence_length(self):
        """Dirty-limit BCS coherence length, in m, from normal-state resistivity."""
        return 1.05e-3 * np.sqrt(self.rho / self.Tc)

    def __str__(self):
        return (
            f"Superconducting metal: Tc = {self.Tc:3.2f} K, rho = {self.rho:3.2e} Ohm*m, "
            f"lambda_L = {self.london_penetration_depth()*1e9:3.1f} nm, "
            f"xi_0 = {self.coherence_length()*1e9:3.1f} nm"
        )
