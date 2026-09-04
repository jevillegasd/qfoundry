"""Superconducting material properties.

Models a thin-film superconductor from its critical temperature, normal-state
resistivity and superconducting carrier density. These properties feed the
BCS superconducting gap (used for Josephson junction Ic/Rn conversions) and
the London penetration depth / coherence length (used for CPW kinetic
inductance calculations).

References
----------
- Tinkham, Introduction to Superconductivity - BCS gap, coherence length and
  the Pippard dirty-limit penetration depth
- Wallraff et al. (2008), arXiv:0807.4094 - CPW kinetic inductance
- Ashcroft & Mermin, Solid State Physics - free-electron Fermi surface
  (Fermi wavevector/velocity) and the Drude relaxation time
- Zmuidzinas, Annu. Rev. Condens. Matter Phys. 3, 169 (2012) - sheet kinetic
  inductance from sheet resistance in the local (dirty) limit,
  L_k/sq = hbar*Rs/(pi*Delta)
"""

import numpy as np
from scipy.constants import m_e, e, hbar
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

    def sc_gap_0(self):
        """Zero-temperature BCS gap Delta_0 = 1.764 kB Tc, in Joules.

        The single source of the BCS weak-coupling prefactor — every
        gap-derived quantity in this class (Delta(T), xi_0, dirty-limit
        L_sq) goes through here so the gap model lives in exactly one place.
        """
        return 1.764 * k_B * self.Tc

    def sc_gap(self):
        """BCS superconducting gap Delta(T) in Joules.

        Delta_0 = 1.764 kB Tc at T → 0 (see :meth:`sc_gap_0`). Above ~0.2·Tc
        the standard BCS interpolation
        Delta(T) = Delta_0 · tanh(1.74·sqrt(Tc/T − 1)) is used (accurate to
        ~2% over the full range; see Tinkham ch. 3). Returns 0 at or above Tc.
        """
        delta_0 = self.sc_gap_0()
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

    def sheet_resistance(self):
        """Normal-state sheet resistance Rs = rho/t, in Ohm/square.

        Requires ``thickness`` to be set.
        """
        if not self.thickness:
            raise ValueError(
                "sheet_resistance requires the film thickness to be set."
            )
        return self.rho / self.thickness

    def fermi_wavevector(self):
        """Fermi wavevector k_F, in 1/m, from the free-electron model.

        k_F = (3 pi^2 n_s)^(1/3), with n_s the superconducting (~normal-state
        conduction) electron density.
        """
        return (3 * np.pi**2 * self.n_s) ** (1 / 3)

    def fermi_velocity(self):
        """Fermi velocity v_F = hbar k_F / m_e, in m/s (free-electron model)."""
        return hbar * self.fermi_wavevector() / m_e

    def scattering_time(self):
        """Drude elastic-scattering time tau, in s, from the normal-state
        resistivity: tau = m_e / (n_s e^2 rho)."""
        return m_e / (self.n_s * e**2 * self.rho)

    def mean_free_path(self):
        """Electron mean free path l = v_F * tau, in m."""
        return self.fermi_velocity() * self.scattering_time()

    def coherence_length_bcs(self):
        """Intrinsic (clean-limit) BCS coherence length xi_0, in m.

        xi_0 = hbar*v_F/(pi*Delta_0) with Delta_0 the zero-temperature gap
        (Tinkham eq. 3.3; the familiar 0.18·hbar·v_F/(kB·Tc) form is this
        with Delta_0 = 1.764 kB Tc substituted in). xi_0 is a T=0-defined
        quantity, hence :meth:`sc_gap_0` rather than the T-dependent gap.
        Unlike :meth:`coherence_length` (an empirical resistivity-based
        dirty-limit estimate), this is derived from the free-electron Fermi
        velocity and does not depend on rho.
        """
        return hbar * self.fermi_velocity() / (np.pi * self.sc_gap_0())

    def effective_penetration_depth(self):
        """Dirty-limit (Pippard-corrected) effective penetration depth, in m.

        lambda_eff = lambda_L * sqrt(xi_0/l), valid for l << xi_0 (mean free
        path much shorter than the intrinsic coherence length).
        """
        return self.london_penetration_depth() * np.sqrt(
            self.coherence_length_bcs() / self.mean_free_path()
        )

    def sheet_kinetic_inductance(self, limit: str = "auto"):
        """Sheet kinetic inductance L_sq, in H/square.

        For the kinetic inductance per unit length of an actual trace
        geometry, use the conformal-mapping result in
        :class:`qfoundry.waveguides.cpw` (``cpw.inductances``) — a bare
        L_sq/W estimate ignores field concentration at the trace edges.

        Parameters
        ----------
        limit : {"auto", "clean", "dirty"}, default "auto"
            "clean" uses the London-depth result L_sq = mu_0 lambda_L^2 / t
            (requires ``thickness``). "dirty" uses the local-limit
            Mattis-Bardeen result L_sq = hbar Rs / (pi Delta(T)), with Rs
            the sheet resistance (also requires ``thickness``, via
            :meth:`sheet_resistance`) and Delta(T) from :meth:`sc_gap`
            (Zmuidzinas 2012). "auto" picks "dirty" when the mean free path
            is shorter than the intrinsic coherence length
            (mean_free_path() < coherence_length_bcs()) and "clean"
            otherwise.
        """
        if limit == "auto":
            limit = "dirty" if self.mean_free_path() < self.coherence_length_bcs() else "clean"
        if limit == "clean":
            if not self.thickness:
                raise ValueError(
                    "sheet_kinetic_inductance('clean') requires the film thickness to be set."
                )
            return mu_0 * self.london_penetration_depth() ** 2 / self.thickness
        if limit == "dirty":
            return hbar * self.sheet_resistance() / (np.pi * self.sc_gap())
        raise ValueError(f"Unknown limit {limit!r}; expected 'auto', 'clean' or 'dirty'.")

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