"""Transmon and tunable transmon models.

References
----------
- Koch et al., Phys. Rev. A 76, 042319 (2007)
- Krantz et al., Appl. Phys. Rev. 6, 021318 (2019) / arXiv:1904.06560
"""

from qfoundry.resonator import cpw, circuit, cpw_resonator
from qfoundry.utils import sc_metal, Ic_to_R, R_to_Ic
import scqubits as scq

from scipy.constants import e as e_0
from scipy.constants import Planck as h_0

from numpy import sqrt, pi, tanh, abs


class transmon(circuit):
    """
    Single Junction Qubit
        R_j:float=0.0,       # Total junction resistance
        E_j:float=0.0,
        C_sum:float=67.5e-15,
        C_g:float  =21.7e-15,
        C_k:float  =36.7e-15,
        C_xy:float =0.e-15,
        C_in:float =8.98e-15,
        res_ro     = cpw_resonator(cpw(11.7,0.1,12,6, alpha=2.4e-2),frequency = 7e9, length_f = 2),    #Readout Resonator
        R_jx:float = 0.0,       # Resistance correction factor
        mat = sc_metal(1.14),
        T = 20.e-3,
        kappa = 0.0,
        ng =0.3 #Offset Charge
        NOTE: Energies are in E/h (not E/hbar)
    """

    def __init__(
        self,
        I_c: float,  # Total junction resistance
        C_sum: float = 67.5e-15,
        C_g: float = 21.7e-15,
        C_k: float = 36.7e-15,
        C_xy: float = 0.0e-15,
        res_ro=cpw_resonator(
            cpw(11.7, 0.1, 12, 6, alpha=2.4e-2), frequency=7e9, length_f=2
        ),  # Readout Resonator
        R_jx: float = 0.0,  # Resistance correction factor
        mat=sc_metal(1.14, 20e-3),
        kappa=0.0,
        ng=0.3,  # Offset Charge
        ncut=40,
        truncated_dim=10,
    ):
        self.mat = mat
        self.R_jx = R_jx
        self.ic = I_c
        self.R_j = Ic_to_R(self.ic, mat=mat, R_jx=R_jx)

        self.C_sum = C_sum
        self.C_g = C_g
        self.C_k = C_k
        self.C_xy = C_xy
        self.Cr = res_ro.C
        self.res_ro = res_ro
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim

        self.qmodel = scq.Transmon(
            EJ=self.Ej() / 1e9 * 2 * pi,
            EC=self.Ec() / 1e9 * 2 * pi,
            ng=ng,
            ncut=ncut,
            truncated_dim=truncated_dim,
        )

        self.Delta = abs(self.res_ro.f0() - self.f01())
        if kappa == 0.0:
            self.kappa = self.res_ro.kappa_ext()
        else:
            self.kappa = kappa
        self._C_ = C_sum

    @classmethod
    def from_ej(cls, E_j: float, **kwargs):
        """
        Initialize transmon from Josephson energy E_j.
        """
        i_c = E_j * 4 * pi * e_0 / h_0
        return cls(I_c=i_c, **kwargs)

    @classmethod
    def from_rj(cls, R_j: float, **kwargs):
        """
        Initialize transmon from junction resistance R_j.
        """
        mat = kwargs.get("mat", sc_metal(1.14, 20e-3))
        r_jx = kwargs.get("R_jx", 0.0)
        i_c = R_to_Ic(R_j - r_jx, mat=mat)
        return cls(I_c=i_c, **kwargs)

    @classmethod
    def from_f01(cls, f01: float, C_sum: float, **kwargs):
        """
        Initialize transmon from a target qubit frequency f01.

        This uses the approximation f01 = sqrt(8 * Ej * Ec) - Ec to find the
        required Ej for a given Ec.
        """
        # Calculate Ec from C_sum
        ec = e_0**2 / (2 * C_sum) / h_0

        # Calculate required Ej from f01 and Ec
        ej = (f01 + ec) ** 2 / (8 * ec)

        # Convert Ej to Ic
        i_c = ej * (2 * e_0 * 2 * pi)

        # Note: C_sum from arguments is passed via kwargs
        return cls(I_c=i_c, C_sum=C_sum, **kwargs)

    def alpha(self):
        """
        Anharmonicity
        """
        return self.qmodel.anharmonicity() * 1e9 / (2 * pi)

    def L(self, phi=0.0):
        """
        RLC circuit modcel josephson inductance for the ground state
        """
        from numpy import cos

        return h_0 / (2 * e_0 * self.Ic()) * 1 / (cos(phi))
        # return (self.f01()*sqrt(self.C()))**-2

    def Ic(self):
        return self.ic

    def Rj(self):
        """
        Total junction resistance
        """
        return self.R_j + self.R_jx

    def Ec(self):
        """
        Capacitive energy
        """
        return e_0**2 / (2 * self.C_sum) / h_0

    def Ej(self):
        """
        Josephson energy
        """
        return self.Ic() / (2 * e_0) / (2 * pi)

    def g01(self):
        """
        Coupling strength between the qubit and the resonator
        """
        w_r = self.res_ro.f0() * 2 * pi
        hbar = h_0 / (2 * pi)
        C_r: cpw_resonator = self.res_ro.C()

        return (
            e_0
            * self.C_g
            / (self.C_g + self.C_sum)
            * sqrt(2 * self.res_ro.f0() / (h_0 * C_r))
        )

    def chi(self):
        """
        Dispersive shift
        https://arxiv.org/pdf/1904.06560 eq. 146
        """
        return (self.g01() ** 2) / (self.Delta) * (1 / (1 + self.Delta / self.alpha()))

    def f01(self):
        """
        Qubit 01 frequency
        """
        return self.qmodel.E01() / (2 * pi) * 1e9
        # return ((8*self.Ej()*self.Ec())**0.5-self.Ec())

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

    def f0(self):
        return self.f01()

    def C(self):
        return self.C_sum

    def T1_max(self):
        """
        Higher bound of T1
        """
        return (self.Delta / self.g01()) ** 2 / (self.kappa)

    def __str__(self):
        return (
            "Ec = \t%3.2f MHz \nEj = \t%3.2f GHz \nIc = \t%3.2f nA \nEJ/EC= \t%1.2f\nf_01 = \t%3.2f GHz \n"
            "f_02 = \t%3.2f GHz \ng_01 = \t%3.2f MHz \nchi =\t%3.2f MHz \nT1_max =\t%3.2f us\n"
            "alpha =\t%3.2f MHz"
            % (
                self.Ec() * 1e-6,
                self.Ej() * 1e-9,
                self.Ic() * 1e9,
                self.Ej() / self.Ec(),
                self.f01() * 1e-9,
                self.f02() * 1e-9,
                self.g01() * 1e-6,
                self.chi() * 1e-6,
                self.T1_max() * 1e6,
                self.alpha() * 1e-6,
            )
        )


class tunable_transmon(transmon):
    """
    Tunable Transmon Qubit
    """

    def __init__(self, flux=0.0, d=1, *args, **kwargs):
        self.flux = flux
        self.d = d  # Assymetry parameter

        super().__init__(*args, **kwargs)

        self.qmodel = scq.TunableTransmon(
            EJmax=self.Ej() / 1e9 * 2 * pi,
            EC=self.Ec() / 1e9 * 2 * pi,
            d=self.d,
            flux=self.flux,
            ng=self.ng,
            ncut=self.ncut,
            truncated_dim=self.truncated_dim,
        )

    def Ej1(self):
        """
        Josephson energy for the first junction
        """
        return self.Ej() * (1 + self.d) / 2

    def Ej2(self):
        """
        Josephson energy for the second junction
        """
        return self.Ej() * (1 - self.d) / 2

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

    def __str__(self):
        return super().__str__() + "\nFlux = \t%3.2f" % self.flux
