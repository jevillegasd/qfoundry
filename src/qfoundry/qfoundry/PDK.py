# Constants and parameters for the PDK
from .utils import sc_metal
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


class PDK:
    """PDK class containing constants and parameters for the design kit.
    Defaults values corresponf to the updated Qfoundry_PDK
    """
    
    def __init__(self):
        """Initialize the PDK with parameters."""
        self.epsilon_r = 11.6883144  # Intrinsic Silicon modified for model

        self.substrate_h = 550e-6  # [μm]
        self.substrate_rho = 1.0e-10  # Substrate conductivity [1/Ω*cm]

        """Coplanar Waveguide parameters"""
        self.cpw_w = 15e-6  # [μm] Waveguide width
        self.cpw_g = 7.5e-6  # [μm] Waveguide gap
        self.cpw_t = 0.1e-6  # [μm] Waveguide thickness
        self.alpha = 3.165e-3  # Superconductive Loss tangent

        """Josephson Junction parameters"""
        self.jj_rhort = 0.535244811537077e-05  # Josephson Junction R.T. resistivity Ohm*m^2
        self.jj_R0 = 4  # R.T. Contact probing correction
        self.jj_rhox = 0  # Josephson Junction resistivity correction (to match measured qubit Ej)
        self.jj_gammax = 4.513e-07  # Josephson Junction Capacitance per unit area F/m^2 (used to match measured qubit Ec)

        """Waveguide and resonator model corrections"""
        self.C_mx = 0  # Waveguide capacitance per unit length correction for waveguides
        self.C_x = 0.81e-15  # Capacitance correction (from measurements modelling) for resonators
        self.C_b = 0.434e-15  # Capacitance per airbridge
        self.C_k = 2.25e-15  # Default Coupling between resonator and feedline.
        self.C_rg = 0.0  # Default Capacitance between resonator and ground plane at the qubit coupling point.

        self.design_rules = {
            "DR_MIN_WAVEGUIDE_WIDTH": DR_MIN_WAVEGUIDE_WIDTH,
            "DR_MIN_WAVEGUIDE_GAP": DR_MIN_WAVEGUIDE_GAP,
            "DR_MIN_WAVEGUIDE_WIDTH": DR_MIN_WAVEGUIDE_WIDTH,
        }
        self.material = "Al"  # Default material
        self.Tc = 1.01  # Critical temperature [K]
        self.mat_prop = sc_metal(self.Tc, 25e-3)  # At 25 mK

    def __str__(self):
        """String representation of the QW_PDK."""
        return f"qfoudnry_PDK(epsilon_r={self.epsilon_r}, substrate_h={self.substrate_h}, substrate_rho={self.substrate_rho}, cpw_w={self.cpw_w}, cpw_g={self.cpw_g}, cpw_t={self.cpw_t}, alpha={self.alpha})"

    def __repr__(self):
        """String representation of the QW_PDK for debugging."""
        return f"qfoudnry_PDK(epsilon_r={self.epsilon_r}, substrate_h={self.substrate_h}, substrate_rho={self.substrate_rho}, cpw_w={self.cpw_w}, cpw_g={self.cpw_g}, cpw_t={self.cpw_t}, alpha={self.alpha})"


class QW_PDK(PDK):
    """QW_PDK class containing constants and parameters for the QW design kit."""

    def __init__(self):
        """Initialize the QW_PDK with parameters from the PDK class."""
        super().__init__()

        # QW_PDK specific parameters
        self.substrate_h = 525e-6  # [μm]
        self.epsilon_r = 12.07  # Intrinsic Silicon modified for model
        self.substrate_rho = 1 / 1e4  # Substrate conductivity [1/Ω*cm]
        self.Lk = 0.0  # Metal layer kinetic inductance [pH//□]
        self.cpw_t = 0.2e-6  # [μm] Waveguide thickness

        DR_MIN_FEATURE_SIZE = DesignRule(
            name="DR_MIN_FEATURE_SIZE",
            description="Minimum junction width thickness",
            value=3e-6,
        )
        DR_DICING_MARKERS_SPACING = DesignRule(
            name="DR_DICING_MARKERS_SPACING",
            description="Spacing between dicing markers",
            value=80e-6,
        )
        self.design_rules["DR_MIN_WAVEGUIDE_WIDTH"] = DR_MIN_FEATURE_SIZE
        self.design_rules["DR_DICING_MARKERS_SPACING"] = DR_DICING_MARKERS_SPACING

    def get_design_rule(self, name):
        """Get a design rule by its name."""
        return self.design_rules.get(name, None)


# Example design rules
