# Constants and parameters for the PDK

epsilon_r = 11.6883144      #Intrinsic Silicon modified for model

substrate_h = 550e-6      #[μm]
substrate_rho = 1.0E-10  #Substrate resistivity [Ω*cm]

'''Coplanar Waveguide parameters'''
cpw_w:float = 15e-6         #[μm] Waveguide width
cpw_g = 7.5e-6        #[μm] Waveguide gap
cpw_t = 0.1e-6         #[μm] Waveguide thickness
alpha = 3.165e-3          # Superconductive Loss tangent

'''Josephson Junction parameters'''
jj_rhort   = 0.535244811537077E-05     # Josephson Junction R.T. resistivity Ohm*m^2
jj_R0      = 4.119856e3                # R.T. Contact probing correction
jj_rhox    = 0                  # Josephson Junction resistivity correction (to match measured qubit Ej)
jj_gammax  = 4.513E-07          # Josephson Junction Capacitance per unit area correction 


# Resonator model corrections
C_mx = 0            # Waveguide capacitance per unit length correction
C_x = 0.81e-15      # Capacitance correction (from measurements modelling)
C_b = 0.434e-15     # Capacitance per airbridge
C_k = 2.25e-15      # Coupling between resonator and feedline.
C_rg = 0.0          # Capacitance between resonator and ground plane at the qubit coupling point.


# rho as a property
def jj_rho():
    return (jj_rhort + jj_rhox)*1e-4 # Convert to Ohm*m^2