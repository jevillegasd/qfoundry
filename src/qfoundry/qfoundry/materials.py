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

def ab_effective_gap(delta1: float, delta2: float) -> float:
    r"""Effective BCS gap of an asymmetric-lead Josephson junction, in J.

    Ambegaokar–Baratoff generalised to leads with different gaps
    (Anderson's formula, T → 0):

        Ic·Rn = \frac{2 \Delta_1 \Delta_2}{e(\Delta_1+\Delta_2)}
                K\!\left(\frac{|\Delta_1-\Delta_2|}{\Delta_1+\Delta_2}\right)

    where K is the complete elliptic integral of the first kind. This
    returns the equivalent gap Δ_eff such that the symmetric relation
    Ic·Rn = πΔ_eff/(2e) reproduces that product:

        Δ_eff = \frac{4 \Delta_1 \Delta_2}{\pi(\Delta_1+\Delta_2)} K(k)

    Equal gaps reduce to Δ_eff = Δ. This is what makes the deliberate lead
    thickness asymmetry (e.g. 30/60 nm Al) enter the junction Ic/Rn model.

    References
    ----------
    - Ambegaokar & Baratoff, Phys. Rev. Lett. 10, 486 (1963) — erratum of
      PRL 11, 104 (1963), asymmetric-gap result.
    """
    from scipy.special import ellipk

    if delta1 <= 0 or delta2 <= 0:
        raise ValueError("ab_effective_gap requires positive gaps.")
    if delta1 == delta2:
        return delta1
    k = abs(delta1 - delta2) / (delta1 + delta2)
    # scipy's ellipk takes the parameter m = k².
    return 4.0 * delta1 * delta2 / (np.pi * (delta1 + delta2)) * ellipk(k**2)


def jj_effective_gap(Tc1: float = None, Tc2: float = None, T: float = 20e-3) -> float:
    """Effective junction gap Δ_eff in J from one or two lead critical
    temperatures.

    ``Tc1`` is the base (bottom) lead, ``Tc2`` the counter (top) lead.
    ``Tc1=None`` falls through to the thin-aluminum default (Tc=1.14 K);
    ``Tc2`` unset/non-positive/equal to ``Tc1`` gives the symmetric BCS gap,
    otherwise the asymmetric-lead Ambegaokar–Baratoff gap
    (:func:`ab_effective_gap`). This is the single source of truth for the
    junction gap used in Ic·Rn / k_Δ relations.
    """
    delta_1 = (sc_metal(Tc=Tc1, T=T) if Tc1 else sc_metal(T=T)).sc_gap()
    if Tc2 and Tc2 > 0 and Tc2 != Tc1:
        return ab_effective_gap(delta_1, sc_metal(Tc=Tc2, T=T).sc_gap())
    return delta_1


Avogadro = 6.022e23  # atoms per mol
Al_mass = 26.98e-3  # kg/mol
Al_density = 2.7e3  # kg/m^3
n_Al = Avogadro * Al_density / Al_mass  # atoms / m^3
mu_0 = 4 * np.pi * 1e-7  # H/m

Nb_mass = 92.906e-3  # kg/mol
Nb_density = 8.57e3  # kg/m^3
n_Nb = Avogadro * Nb_density / Nb_mass  # atoms / m^3

Ta_mass = 180.95e-3  # kg/mol
Ta_density = 16.69e3  # kg/m^3
n_Ta = Avogadro * Ta_density / Ta_mass  # atoms / m^

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
    name : str, optional
        Human-readable label (e.g. "Al 30nm", "Nb/Ta stack"). Two films of
        the same element but different thickness are distinct materials —
        their gaps differ, which matters e.g. for quasiparticle trapping in
        asymmetric junction leads.
    thickness : float, optional
        Film thickness in m. Purely descriptive on sc_metal (geometry users
        like cpw take thickness explicitly); sc_stack computes it from its
        layers.

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
        name: str = None,
        thickness: float = None,
    ):
        self.Tc = Tc
        self.T = T
        self.rho = rho
        self.n_s = n_s
        self.name = name
        self.thickness = thickness

    def sc_gap(self):
        """BCS superconducting gap Delta(T) in Joules.

        Delta_0 = 1.764 kB Tc at T → 0. Above ~0.2·Tc the standard BCS
        interpolation Delta(T) = Delta_0 · tanh(1.74·sqrt(Tc/T − 1)) is used
        (accurate to ~2% over the full range; see Tinkham ch. 3). Returns 0
        at or above Tc.
        """
        delta_0 = 1.764 * k_B * self.Tc
        if self.T <= 0.2 * self.Tc:
            return delta_0
        if self.T >= self.Tc:
            return 0.0
        return delta_0 * np.tanh(1.74 * np.sqrt(self.Tc / self.T - 1))

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


class sc_stack(sc_metal):
    r"""
    Effective superconducting material for a thin-film stack (bilayer or
    multilayer), combining constituent films via a simplified proximity-
    effect model.

    Parameters
    ----------
    layers : list of (sc_metal, float)
        (material, thickness [m]) pairs, in stacking order. Each material's
        own operating temperature is ignored; the stack's T (below) is used
        throughout.
    T : float, default=20e-3
        Operating temperature of the stack, in K.

    Attributes
    ----------
    thickness : float
        Total stack thickness (sum of layer thicknesses), in m. Convenient
        for passing straight to ``cpw(thickness=stack.thickness, ...)``.

    Notes
    -----
    Assumes each film is thin compared to its coherence length, so a single
    spatially-uniform effective order parameter describes the whole stack
    (i.e. the "dirty limit" approximation). The effective Tc, rho and n_s
    are calculated as weighted averages of the constituent layers, weighted by
    their thicknesses and superconducting carrier densities. The effective
    London penetration depth and coherence length are then derived from these
    effective parameters.

    References
    ----------
    - Cooper, Phys. Rev. Lett. 6, 689 (1961) - S-N proximity effect
    - McMillan, Phys. Rev. 175, 537 (1968) - tunneling model for proximity bilayers
    """

    def __init__(self, layers, T: float = 20e-3, name: str = None):
        if not layers:
            raise ValueError("sc_stack requires at least one (material, thickness) layer.")

        self.layers = list(layers)
        thickness = sum(d for _, d in self.layers)
        if thickness <= 0:
            raise ValueError("sc_stack layer thicknesses must sum to a positive value.")

        dos_weight = sum(d * m.n_s for m, d in self.layers)
        Tc_eff = sum(d * m.n_s * m.Tc for m, d in self.layers) / dos_weight
        n_s_eff = dos_weight / thickness
        rho_eff = thickness / sum(d / m.rho for m, d in self.layers)

        super().__init__(Tc=Tc_eff, T=T, rho=rho_eff, n_s=n_s_eff,
                         name=name, thickness=thickness)

    def __str__(self):
        stack = " / ".join(f"{d*1e9:3.1f} nm (Tc={m.Tc:3.2f} K)" for m, d in self.layers)
        return f"Superconducting stack [{stack}]: {super().__str__()}"


mat_al = sc_metal(Tc=1.14, T=20e-3, rho=2.06e-9, n_s=3 * n_Al, name="Al")
mat_nb = sc_metal(Tc=9.2, T=20e-3, rho=1.5e-7, n_s=n_Nb, name="Nb")
mat_ta = sc_metal(Tc=4.5, T=20e-3, rho=1.5e-7, n_s=n_Ta, name="Ta")
mat_nb_ta = sc_stack([(mat_nb, 100e-9), (mat_ta, 10e-9)], T=20e-3, name="Nb/Ta")