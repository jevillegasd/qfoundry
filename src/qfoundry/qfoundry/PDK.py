# Constants and parameters for the PDK
from math import pi
from scipy.constants import elementary_charge as _e0
from qfoundry.materials import sc_metal, sc_stack, mat_nb, mat_ta, n_Al
from qfoundry.waveguides import cpw

class DesignRule:
    """Design rule class containing constants and parameters for the design rules."""

    def __init__(self, name, description, value):
        """Initialize the DesignRule with parameters."""
        self.name = name
        self.description = description
        self.value = value

    def __str__(self):
        """String representation of the DesignRule."""
        return f"{self.name}: {self.description} = {self.value}"

    def __repr__(self):
        """String representation of the DesignRule for debugging."""
        return f"DesignRule(name={self.name}, description={self.description}, value={self.value})"


## Deafult design rules
DR_MIN_WAVEGUIDE_WIDTH = DesignRule(
    name="DR_MIN_WAVEGUIDE_WIDTH", description="Minimum waveguide width", value=3e-6
)

DR_MIN_WAVEGUIDE_GAP = DesignRule(
    name="DR_MIN_WAVEGUIDE_GAP", description="Minimum waveguide gap", value=1e-6
)

DR_MIN_JUCNTION_WIDTH = DesignRule(
    name="DR_MIN_JUCNTION_WIDTH",
    description="Minimum jucntion width tckness",
    value=100e-9,
)

DR_MIN_JUCNTION_CURRENT = DesignRule(
    name="DR_MIN_JUCNTION_CURRENT", description="Minimum jucntion current", value=9.0e-9
)

DR_MAX_JUCNTION_CURRENT = DesignRule(
    name="DR_MAX_JUCNTION_CURRENT", description="Maximum jucntion current", value=90e-9
)

DR_DICING_MARKERS_SPACING = DesignRule(
    name="DR_DICING_MARKERS_SPACING",
    description="Spacing between dicing markers",
    value=80e-6,
)


class WaveguideSpec:
    """Named waveguide definition inside a PDK.

    A PDK may define several waveguides (readout CPW, feedline CPW, a
    high-kinetic-inductance nanowire for super-inductors, …), each with its
    own geometry and an assigned material from the PDK's material registry.

    Parameters
    ----------
    name : str
        Registry key (e.g. "cpw", "feedline", "nanowire").
    width : float
        Center conductor width in m.
    spacing : float
        Gap to ground plane in m.
    thickness : float, optional
        Metal thickness in m. Falls back to the assigned material's film
        thickness when None.
    material : str, default="base"
        Name of a material in the PDK's ``materials`` registry.
    alpha : float, default=2.4e-2
        Attenuation coefficient in m^-1.
    wg_type : str, default="cpw"
        Waveguide topology ("cpw", "microstrip", "nanowire", …). Only "cpw"
        currently has a builder (PDK.cpw()).
    """

    def __init__(self, name, width, spacing, thickness=None, material="base",
                 alpha=2.4e-2, wg_type="cpw"):
        self.name = name
        self.width = width
        self.spacing = spacing
        self.thickness = thickness
        self.material = material
        self.alpha = alpha
        self.wg_type = wg_type

    def __repr__(self):
        return (f"WaveguideSpec(name={self.name!r}, wg_type={self.wg_type!r}, "
                f"width={self.width}, spacing={self.spacing}, "
                f"thickness={self.thickness}, material={self.material!r}, alpha={self.alpha})")


class PDK:
    """Process design kit: substrate, materials, waveguides and junction stack.

    A PDK carries a registry of superconducting ``materials`` (films of the
    same element at different thicknesses count as different materials — the
    gap asymmetry between a 30 nm and a 60 nm lead suppresses quasiparticle
    exchange across the junction) and a registry of ``waveguides``, each of
    which is assigned one of those materials. Josephson junctions reference
    two lead materials (base/bottom and counter/top electrode).

    The legacy flat attributes (``cpw_w``, ``cpw_g``, ``cpw_t``, ``alpha``,
    ``Tc``, ``metal_rho``, ``metal_n_s``, ``mat_prop``) are kept as
    properties that read/write the default waveguide and *its* material, so
    existing single-material code keeps working unchanged — including when
    the default waveguide is re-assigned to a different metal (see
    ``QF_NbTa_PDK``).
    """

    def __init__(self, name: str = "default"):
        """Initialize the PDK with parameters."""
        self.name = name
        self.epsilon_r = 11.6883144  # Intrinsic Silicon modified for model

        self.substrate_h = 550e-6  # [μm]
        self.substrate_rho = 1.0e-10  # Substrate conductivity [1/Ω*cm]

        self.T_op = 0.02  # Operating temperature [K]

        # ── Material registry ────────────────────────────────────────────
        # "base" starts as the ground-plane / waveguide metal (thin aluminum
        # film); the legacy Tc/metal_rho/metal_n_s properties delegate to
        # whichever material the default waveguide is assigned.
        self.materials: dict[str, sc_metal] = {
            "base": sc_metal(
                Tc=1.14, T=self.T_op, rho=2.06e-9, n_s=3 * n_Al,
                name="base", thickness=0.1e-6,
            ),
        }

        # ── Waveguide registry ───────────────────────────────────────────
        # Legacy cpw_w/cpw_g/cpw_t/alpha properties delegate to the default.
        self.waveguides: dict[str, WaveguideSpec] = {
            "cpw": WaveguideSpec(
                name="cpw", width=15e-6, spacing=7.5e-6, thickness=0.1e-6,
                material="base", alpha=3.165e-3, wg_type="cpw",
            ),
        }
        self.default_waveguide = "cpw"

        """Josephson Junction parameters"""
        self.jj_rhort = (
            0.535244811537077e-05  # Josephson Junction R.T. resistivity Ohm*m^2
        )
        # R.T. probing correction Rx [Ω]. Positive *additive* convention:
        # the raw probe reading understates the junction — effective junction
        # resistance in the AB product = raw reading + jj_R0 (empirically
        # fitted; candidate causes: RT parallel conduction through the
        # substrate/barrier traps that freezes out cold, probe→cooldown aging).
        self.jj_R0 = 4.119856e3
        self.jj_rhox = (
            0  # Josephson Junction resistivity correction (to match measured qubit Ej)
        )
        self.jj_gammax = (
            4.513e-07  # Josephson Junction Capacitance per unit area correction
        )
        self.RI_factor: float | None = None  # Measured Ic·(Rn+Rx) AB product [V]
        # Junction lead materials (names in self.materials); None falls back
        # to the "base" material. Base = bottom electrode, counter = top.
        self.jj_base_material: str | None = None
        self.jj_counter_material: str | None = None

        # Resonator model corrections
        self.C_mx = 0  # Waveguide capacitance per unit length correction
        self.C_x = 0.81e-15  # Capacitance correction (from measurements modelling)
        self.C_b = 0.434e-15  # Capacitance per airbridge
        self.C_k = 2.25e-15  # Coupling between resonator and feedline.
        self.C_rg = 0.0  # Capacitance between resonator and ground plane at the qubit coupling point.

        self.design_rules = {
            "DR_MIN_WAVEGUIDE_WIDTH": DR_MIN_WAVEGUIDE_WIDTH,
            "DR_MIN_WAVEGUIDE_GAP": DR_MIN_WAVEGUIDE_GAP,
        }

    # ── Registry accessors ────────────────────────────────────────────────

    def material(self, name: str | None = None) -> sc_metal:
        """Return a material by name; None returns the default waveguide's
        material (the legacy single-material view of the PDK)."""
        key = name if name is not None else self.waveguide().material
        if key not in self.materials:
            raise KeyError(
                f"Material {key!r} not defined in PDK {self.name!r}; "
                f"available: {list(self.materials)}"
            )
        return self.materials[key]

    def add_material(self, material: sc_metal, name: str | None = None) -> sc_metal:
        """Register a material under ``name`` (defaults to material.name)."""
        key = name or getattr(material, "name", None)
        if not key:
            raise ValueError("Material needs a name (pass name= or set material.name).")
        material.name = key
        self.materials[key] = material
        return material

    def waveguide(self, name: str | None = None) -> WaveguideSpec:
        """Return a waveguide spec by name; None returns the default one."""
        key = name or self.default_waveguide
        if key not in self.waveguides:
            raise KeyError(
                f"Waveguide {key!r} not defined in PDK {self.name!r}; "
                f"available: {list(self.waveguides)}"
            )
        return self.waveguides[key]

    def add_waveguide(self, spec: WaveguideSpec, default: bool = False) -> WaveguideSpec:
        """Register a waveguide spec; optionally make it the default."""
        if spec.material not in self.materials:
            raise KeyError(
                f"Waveguide {spec.name!r} references unknown material {spec.material!r}; "
                f"available: {list(self.materials)}"
            )
        self.waveguides[spec.name] = spec
        if default:
            self.default_waveguide = spec.name
        return spec

    def jj_lead_materials(self) -> tuple[sc_metal, sc_metal]:
        """Return the (base/bottom, counter/top) junction lead materials.

        Unassigned leads fall back to the default waveguide's material
        (matching the legacy single-metal assumption).
        """
        return (
            self.material(self.jj_base_material),
            self.material(self.jj_counter_material),
        )

    # ── Legacy flat attributes (delegate to base material / default wg) ──

    @property
    def mat_prop(self) -> sc_metal:
        """Material of the default waveguide (legacy single-material view)."""
        return self.material(self.waveguide().material)

    @mat_prop.setter
    def mat_prop(self, material: sc_metal) -> None:
        self.materials[self.waveguide().material] = material

    @property
    def Tc(self) -> float:
        return self.mat_prop.Tc

    @Tc.setter
    def Tc(self, value: float) -> None:
        self.mat_prop.Tc = value

    @property
    def metal_rho(self) -> float:
        return self.mat_prop.rho

    @metal_rho.setter
    def metal_rho(self, value: float) -> None:
        self.mat_prop.rho = value

    @property
    def metal_n_s(self) -> float:
        return self.mat_prop.n_s

    @metal_n_s.setter
    def metal_n_s(self, value: float) -> None:
        self.mat_prop.n_s = value

    @property
    def cpw_w(self) -> float:
        return self.waveguide().width

    @cpw_w.setter
    def cpw_w(self, value: float) -> None:
        self.waveguide().width = value

    @property
    def cpw_g(self) -> float:
        return self.waveguide().spacing

    @cpw_g.setter
    def cpw_g(self, value: float) -> None:
        self.waveguide().spacing = value

    @property
    def cpw_t(self) -> float:
        wg = self.waveguide()
        if wg.thickness is not None:
            return wg.thickness
        return self.material(wg.material).thickness

    @cpw_t.setter
    def cpw_t(self, value: float) -> None:
        self.waveguide().thickness = value

    @property
    def alpha(self) -> float:
        return self.waveguide().alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.waveguide().alpha = value

    def jj_gap(self) -> float:
        """Effective junction gap Δ_eff in J, from the two lead materials.

        Uses the asymmetric-lead Ambegaokar–Baratoff result (see
        qfoundry.materials.ab_effective_gap) with each lead's own BCS gap —
        so the deliberate 30/60 nm lead thickness asymmetry enters the
        Ic·Rn model. Unassigned leads fall back to the default waveguide's
        material, reducing to the classic symmetric πΔ/(2e).
        """
        from qfoundry.materials import ab_effective_gap
        base, counter = self.jj_lead_materials()
        return ab_effective_gap(base.sc_gap(), counter.sc_gap())

    @property
    def k_Delta(self) -> float | None:
        """Modified Ambegaokar-Baratoff correction factor.

        Defined by:  Ic·(Rn + Rx) = k_Δ · πΔ_eff / (2e)
        where Δ_eff is the effective junction gap from the two lead
        materials (jj_gap(); equal-gap leads give Δ_eff = 1.764·kB·Tc).

        Returns None if RI_factor is not set.
        """
        if self.RI_factor is None or self.RI_factor <= 0:
            return None
        try:
            delta_eff = self.jj_gap()  # J
        except (KeyError, ValueError):
            return None
        return self.RI_factor * 2.0 * _e0 / (pi * delta_eff)

    def Rn_from_Ej(self, Ej_Hz: float) -> float | None:
        """Estimate the raw probe resistance target from design Josephson energy.

        Uses the AB relation:  Rn_calc = RI_factor / Ic_AB - Rx
        where  Ic_AB = Ej · 4πe  (Ej in the E/h "Hz" convention, see
        qfoundry.utils.Ej_to_Ic — no extra /h) and Rx = jj_R0.

        Convention: the stored/returned Rn is the *raw room-temperature probe
        reading*; the effective junction resistance in the AB product is
        Rn + Rx (Rx is a positive additive correction — see jj_R0).

        Returns None if RI_factor is not set or result would be non-positive.
        """
        from qfoundry.utils import Ej_to_Ic
        if self.RI_factor is None or self.RI_factor <= 0 or Ej_Hz <= 0:
            return None
        Ic_AB = Ej_to_Ic(Ej_Hz)
        Rn = self.RI_factor / Ic_AB - self.jj_R0
        return Rn if Rn > 0 else None

    def cpw(self, name: str | None = None):
        """Return a coplanar waveguide object for the named waveguide spec.

        ``name=None`` uses the default waveguide (legacy behavior).
        """
        wg = self.waveguide(name)
        material = self.material(wg.material)
        thickness = wg.thickness if wg.thickness is not None else material.thickness
        return cpw(
            epsilon_r=self.epsilon_r,
            height=self.substrate_h,
            width=wg.width,
            spacing=wg.spacing,
            thickness=thickness,
            material=material,
            alpha=wg.alpha,
        )

    def __str__(self):
        """String representation of the PDK."""
        return f"PDK(name={self.name!r}, epsilon_r={self.epsilon_r}, substrate_h={self.substrate_h}, cpw_w={self.cpw_w}, cpw_g={self.cpw_g}, cpw_t={self.cpw_t}, alpha={self.alpha})"

    def __repr__(self):
        """String representation of the PDK for debugging."""
        return f"PDK(name={self.name!r}, epsilon_r={self.epsilon_r}, substrate_h={self.substrate_h}, cpw_w={self.cpw_w}, cpw_g={self.cpw_g}, cpw_t={self.cpw_t}, alpha={self.alpha})"


# Josephson junction lead films, shared by the QF process lines. Thin Al
# films show a thickness-dependent Tc enhancement, so the 30 nm base and
# 60 nm counter electrode carry different BCS gaps — that asymmetry detunes
# the two leads' quasiparticle spectra, suppressing phonon-mediated
# quasiparticle transfer across the junction, and enters the junction
# Ic·Rn model via the asymmetric Ambegaokar–Baratoff relation
# (qfoundry.materials.ab_effective_gap / PDK.jj_gap()). The Tc values are
# nominal thin-film literature numbers — replace with measured film data.
def _al_jj_leads(T_op: float) -> tuple[sc_metal, sc_metal]:
    return (
        sc_metal(Tc=1.3, T=T_op, rho=2.06e-9, n_s=3 * n_Al,
                 name="Al 30nm", thickness=30e-9),
        sc_metal(Tc=1.2, T=T_op, rho=2.06e-9, n_s=3 * n_Al,
                 name="Al 60nm", thickness=60e-9),
    )


# ── QF_PDK: all-aluminum process line ─────────────────────────────────────
# Circuit layer ("base") and both Josephson junction leads are aluminum,
# with asymmetric lead thicknesses (30 nm base / 60 nm counter electrode).
qf_pdk = PDK(name="QF_PDK")
for _m in _al_jj_leads(qf_pdk.T_op):
    qf_pdk.add_material(_m)
qf_pdk.jj_base_material = "Al 30nm"
qf_pdk.jj_counter_material = "Al 60nm"

# ── QF_NbTa_PDK: Nb/Ta bilayer circuit layer, Al junctions ────────────────
# Resonators and qubit capacitors are Nb capped with Ta (modeled as an
# effective proximity bilayer, see qfoundry.materials.sc_stack); junctions
# remain Al with the same asymmetric 30/60 nm leads. The default waveguide
# is assigned the bilayer, so the legacy flat view (Tc, metal_rho,
# metal_n_s, cpw_t, mat_prop) reflects the Nb/Ta stack.
qf_nbta_pdk = PDK(name="QF_NbTa_PDK")
qf_nbta_pdk.add_material(sc_stack([(mat_nb, 100e-9), (mat_ta, 10e-9)],
                                  T=qf_nbta_pdk.T_op, name="Nb/Ta"))
for _m in _al_jj_leads(qf_nbta_pdk.T_op):
    qf_nbta_pdk.add_material(_m)
qf_nbta_pdk.waveguide().material = "Nb/Ta"
qf_nbta_pdk.waveguide().thickness = None  # fall through to the stack's 110 nm
qf_nbta_pdk.jj_base_material = "Al 30nm"
qf_nbta_pdk.jj_counter_material = "Al 60nm"
del qf_nbta_pdk.materials["base"]  # the bilayer *is* the base metal here

qw_pdk = PDK(name="QW_PDK")  # PDK instance with QW-specific parameters
qw_pdk.substrate_h = 525e-6       # [μm]
qw_pdk.epsilon_r = 12.07          # Intrinsic Silicon modified for model
qw_pdk.substrate_rho = 1 / 1e4   # Substrate conductivity [1/Ω*cm]
qw_pdk.Lk = 0.0                   # Metal layer kinetic inductance [pH//□]
qw_pdk.cpw_t = 0.2e-6             # [μm] Waveguide thickness
qw_pdk.alpha = 0.0027e-3          # Superconductive Loss tangent (np/m)
qw_pdk.design_rules["DR_MIN_WAVEGUIDE_WIDTH"] = DesignRule(
    name="DR_MIN_FEATURE_SIZE",
    description="Minimum junction width thickness",
    value=3e-6,
)
qw_pdk.design_rules["DR_DICING_MARKERS_SPACING"] = DesignRule(
    name="DR_DICING_MARKERS_SPACING",
    description="Spacing between dicing markers",
    value=80e-6,
)




# Registry of all available PDK instances
PDK_REGISTRY: dict[str, PDK] = {
    qf_pdk.name: qf_pdk,
    qf_nbta_pdk.name: qf_nbta_pdk,
    qw_pdk.name: qw_pdk,
}
