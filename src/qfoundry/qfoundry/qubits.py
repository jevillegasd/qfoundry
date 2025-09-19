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
from numpy import diag, ones, arange
from scipy.linalg import eigh
import inspect

class transmon(circuit):
    """
    Single Junction Qubit
        E_j:    float               # Josephson energy
        C_sum:  float=67.5e-15,
        C_g:    float  =21.7e-15,
        C_k:    float  =36.7e-15,
        C_xy:   float =0.e-15,
        res_ro = cpw_resonator(cpw(11.7,0.1,12,6, alpha=2.4e-2),frequency = 7e9, length_f = 2),    #Readout Resonator
        R_jx:   float = 0.0,       # Resistance correction factor
        mat =   sc_metal(1.14),
        T =     20.e-3,
        kappa = 0.0,
        ng = 0.3 #Offset Charge
        NOTE: Energies are in E/h (not E/hbar)
    """
    qmodel = None
    _Rj_ = None  # Junction resistance
    _Rx_ = None  # Junction resistance correction factor

    def __init__(
        self,
        E_j: float,  # Josephson energy
        E_c: float = None,
        C_g: float = 21.7e-15,
        C_xy: float = 0.0e-15,
        res_ro=cpw_resonator(
            cpw(11.7, 0.1, 12, 6, alpha=2.4e-2), frequency=7e9, length_f=4 # Readout Resonator
        ),  
        mat=sc_metal(1.14, 20e-3),
        inst_model = True,  # Whether to instantiate the scquibits model
        **kwargs
    ):
        self.mat = mat
        self._ej_ = E_j
        self._ec_ = E_c   

        if E_c is None:
            C_sum = kwargs.get("C_sum", None)
            assert C_sum is not None, "Either E_c or C_sum should not be provided."
            self._ec_ = e_0**2 / (2 * C_sum) / h_0

        # Resistance values may be provided directly but not used for calculations unless the object
        # is instantiated using from_rj() method.
        
        if self._Rx_ is None:
            self._Rx_ = kwargs.get("R_jx", 0)  

        if self._Rj_ is None:
            self._Rj_ = kwargs.get("R_j", 0)

        self.C_g = C_g
        self.C_xy = C_xy
        self.res_ro = res_ro
        self.kappa = kwargs.get("kappa", self.res_ro.kappa_ext()) # External coupling rate

        # scqubits parameters
        self.ng = kwargs.get("ng", 0.3)     # Offset charge
        self.ncut = kwargs.get("ncut", 8)  # Number of states to truncate the Hilbert space
        self.truncated_dim = kwargs.get("truncated_dim", 8) # Number of states to truncate the Hilbert space

        # Instantiate the scqubits model
        if inst_model:
            self.qmodel = scq.Transmon(
                EJ=self.Ej() / 1e9,
                EC=self.Ec() / 1e9,
                ng=self.ng,
                ncut=self.ncut,
                truncated_dim=self.truncated_dim,
            )
 
        # super (circuit) parameters 
        self._C_ = e_0**2 / (2 * self._ec_) / h_0
        self._L_ = h_0 / (2 * e_0 * self.Ic())
        self._R_ = self._Rj_ + self._Rx_
    @classmethod
    def from_Csum(cls, C_sum: float, **kwargs):
        """
        Initialize transmon from total capacitance C_sum.
        """
        ec = e_0**2 / (2 * C_sum) / h_0 # https://arxiv.org/pdf/cond-mat/0703002 eq. 2.1*
        ej = kwargs.get("E_j", 0.0)
        return cls(E_j=ej, E_c=ec, **kwargs)

    @classmethod
    def from_ic(cls, i_c: float, **kwargs):
        """
        Initialize transmon from critical current I_c.
        """
        E_j = i_c / (2 * e_0 * 2 * pi)
        self._Rx_ = kwargs.get("R_jx", 0.0)
        self._Rj_ = Ic_to_R(i_c, mat=kwargs.get("mat", sc_metal(1.14, 25e-3))) - self._Rx_
        kwargs["R_j"] = self._Rj_
        kwargs["R_jx"] = self._Rx_
        return cls(E_j=E_j, **kwargs)

    @classmethod
    def from_rj(cls, R_j: float, **kwargs):
        """
        Initialize transmon from junction resistance R_j.
        The saved junction resistance is R_j, but the effective resistance 
        used to calculate the qubit properties is R_j - R_jx.
        """
        cls._Rx_ = kwargs.get("R_jx", 0.0)
        cls._Rj_ = R_j

        mat = kwargs.get("mat", sc_metal(1.14, 25e-3))

        E_c = kwargs.get("E_c", None)
        if E_c is None:
            C_sum = kwargs.get("C_sum", None)
            assert C_sum is not None, "C_sum must be provided if E_c is not."
            E_c = e_0**2 / (2 * C_sum) / h_0

        i_c = R_to_Ic(cls._Rj_ + cls._Rx_, mat=mat)
        E_j = i_c / (2 * e_0 * 2 * pi)
        kwargs["E_c"] = E_c
        kwargs["E_j"] = E_j
        return cls(**kwargs)

    @classmethod
    def from_f01(cls, f01: float, **kwargs):
        """
        Initialize transmon from a target qubit frequency f01.

        This uses the approximation f01 = sqrt(8 * Ej * Ec) - Ec to find the
        required Ej for a given Ec.
        """
        cls._Rx_ = kwargs.get("R_jx", 0.0)
        cls._Rj_ = kwargs.get("R_j", None)
        
        E_c = kwargs.get("E_c", None)
        if E_c is None:
            C_sum = kwargs.get("C_sum", None)
            assert C_sum is not None, "C_sum must be provided if E_c is not."
            E_c = e_0**2 / (2 * C_sum) / h_0

        # Calculate Ec from C_sum
        ec = e_0**2 / (2 * C_sum) / h_0

        # Calculate required Ej from f01 and Ec
        ej = (f01 + ec) ** 2 / (8 * ec)
        ic = ej * 2 * e_0 * (2 * pi)

        if cls._Rj_ is None: # The resistance value is only calculated if not provided
            cls._Rj_ = Ic_to_R(ic, mat=kwargs.get("mat", sc_metal(1.14, 25e-3))) - cls._Rx_

        kwargs["R_j"] = cls._Rj_
        kwargs["R_jx"] = cls._Rx_
        return cls(E_j=ej, E_c=ec, **kwargs)

    def alpha(self):
        """
        Anharmonicity
        """
        if self.qmodel is None:
            return -self.Ec() * 1e9
        return self.qmodel.anharmonicity() * 1e9

    def L(self, phi=0.0):
        """
        RLC circuit modcel josephson inductance for the ground state
        """
        from numpy import cos

        return h_0 / (2 * e_0 * self.Ic()) * 1 / (cos(phi))
        # return (self.f01()*sqrt(self.C()))**-2

    def Ic(self):
        return self._ej_ * 2 * e_0 * (2 * pi)

    def Ec(self):
        """
        Capacitive energy
        """
        return self._ec_

    def Ej(self):
        """
        Josephson energy
        """
        return self._ej_

    def nj(self,j):
        """
        Charge matrix element between states j+1 and j
        |< j + 1 | nˆ | j >| ≈
        https://arxiv.org/pdf/cond-mat/0703002 eq 3.4
        """
        n_j_annalytical = np.sqrt((1+j)/2)*(self.Ej() / (8*self.Ec()))**(1/4)
        return n_j_annalytical
    
    def g(self,j):
        """
        Coupling sytrength between states j and j+1
        """
        nj = self.nj(j)

        C_sum = self.C()
        f_r     = self.res_ro.f0() # Resonator angular frequency
        C_r     = self.res_ro.C() # Resonator capacitance
        beta    = self.C_g / (C_sum) # Participation ratio
        V_zpf    = sqrt(h_0 * f_r / (2 * C_r)) # Zero point fluctuation voltage

        g_j = beta * V_zpf * nj * e_0/h_0
        return g_j
    
    def g01(self):
        """
        Coupling strength between the qubit and the resonator (capacitive)
        https://arxiv.org/pdf/cond-mat/0703002 eq. 3.3

        g01 ~ 2 * beta * e * Vrms * n01 / h
        where beta is the participation ratio (beta ~= C_g/ (C_sum)) [coupling capaictance over total capacitance]
        e is the elementary charge,
        Vrms is the root mean square voltage across the resonator, and hbar is the reduced Planck's constant.
        The Vrms can be calculated from the resonator frequency and its capacitance as
        V_rms = sqrt(hbar * omega_r / (2 * C_r))
        
        https://arxiv.org/pdf/cond-mat/0703002 eq. 3.1*
        """
        C_sum = self.C()
        f_r     = self.res_ro.f0() # Resonator angular frequency
        C_r     = self.res_ro.C() # Resonator capacitance
        V_zpf    = sqrt(h_0 * f_r / (2 * C_r)) # Zero point fluctuation voltage
        beta    = self.C_g / (C_sum) # Participation ratio
        return 2* beta * e_0 * V_zpf * self.n01() / h_0  # in Hz
        # return 2 * beta * e_0 * V_zpf * self.n_matrix()[0,1] / h_0 # Numerical alternative for n01

    def n01(self):
        """
        Estimated first charge matrix element between the ground and first excited state
        """
        n01 = (self.Ej()/(8*self.Ec()))**0.25 * 1/ sqrt(2)  # https://arxiv.org/pdf/cond-mat/0703002 eq 3.4
        return n01
    
    def n_matrix(self, N=20, nlevels=10):
        """
        Charge matrix of the qubit
        N: ncharge basis truncation
        """
        ec = self.Ec()
        ej = self.Ej()
        n = arange(-N, N+1)
        H = diag(4 * ec * n**2) - 0.5 * ej * (diag(ones(2*N), 1) + diag(ones(2*N), -1))
        eigvals, eigvecs = eigh(H)
        n_op = diag(n)
        n_matrix = eigvecs.T @ n_op @ eigvecs

        return n_matrix[:nlevels, :nlevels], eigvals[:nlevels] # Energies same units as Ec and Ej (Hz)

    def chi(self):
        """
        Dispersive shift
        https://arxiv.org/pdf/1904.06560 eq. 146
        """
        return -(self.g01() ** 2) / (self.delta()) * (1 / (1 + self.delta() / self.alpha()))

    def f01(self):
        """
        Qubit 01 frequency
        """
        if self.qmodel is None:
            return (sqrt(8*self.Ej()*self.Ec())**0.5-self.Ec())
        return self.qmodel.E01() * 1e9


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
    
    def delta(self):
        """
        Frequency detuning
        """
        return abs(self.res_ro.f0() - self.f01())

    def T1_max(self):
        """
        Higher bound of T1
        """
        return (self.delta() / self.g01()) ** 2 / (self.kappa)

    def __str__(self):
        return (
            "Ec = \t%3.2f MHz \nEj = \t%3.2f GHz \nIc = \t%3.2f nA \nEJ/EC= \t%1.2f\nf_01 = \t%3.2f GHz \n"
            "f_02 = \t%3.2f GHz \ng_01 = \t%3.2f MHz \nchi =\t%3.2f MHz \nT1_max =\t%3.2f us\n"
            "alpha =\t%3.2f MHz"
            % (
                self._ec_ * 1e-6,
                self._ej_ * 1e-9,
                self.Ic() * 1e9,
                self._ej_ / self._ec_,
                self.f01() * 1e-9,
                self.f02() * 1e-9,
                self.g01() * 1e-6,
                self.chi() * 1e-6,
                self.T1_max() * 1e6,
                self.alpha() * 1e-6,
            )
        )
    
    # print
    def __repr__(self):
        return super().__repr__() + self.__str__()


class tunable_transmon(transmon):
    """
    Tunable Transmon Qubit
    """

    def __init__(self, flux=0.0, d=1, *args, **kwargs):
        self.flux = flux
        self.d = d  # Assymetry parameter
        self.ng = kwargs.get("ng", 0.3)  # Offset charge
        if "Ej_max" in kwargs:
            self.Ej_max = kwargs.get("Ej_max", 0.0)  # Maximum Josephson energy
            kwargs.pop("Ej_max")
            kwargs["E_j"] = self.Ej_max
        super().__init__(inst_model=False, *args, **kwargs)

        # Check relevant args for TunableTransmon available in kwargs
        for key, value in kwargs.items():
            if key in [
                "ncut",
                "truncated_dim",
            ]:
                setattr(self, key, value)

        self.qmodel = scq.TunableTransmon(
            EJmax=self.Ej() / 1e9,
            EC=self.Ec() / 1e9,
            d=self.d,
            flux=self.flux,
            ng=self.ng,
            ncut=self.ncut,
            truncated_dim=self.truncated_dim,
        )

    @classmethod # Overrides the parent class method (but is doing the same)
    def from_ic(cls, i_c: float, **kwargs):
        """
        Initialize tunable transmon from critical current I_c.
        """
        E_j = i_c / (2 * e_0 * 2 * pi)
        cls._Rx_ = kwargs.get("R_jx", 0.0)
        cls._Rj_ = Ic_to_R(i_c, mat=kwargs.get("mat", sc_metal(1.14, 25e-3))) - cls._Rx_
        kwargs["R_j"] = cls._Rj_
        kwargs["R_jx"] = cls._Rx_
        return cls(E_j=E_j, **kwargs)
    
    def Ej1(self):
        """
        Josephson energy for the first junction
        """
        return self._ej_ * (1 + self.d) / 2

    def Ej2(self):
        """
        Josephson energy for the second junction
        """
        return self._ej_ * (1 - self.d) / 2

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
    
    def f01(self, flux=None):
        """
        Qubit 01 frequency
        """
        if flux is None:
            flux = self.flux
        if self.qmodel is not None:
            # By default return the f01 using scqubits
            qmodel: scq.TunableTransmon = self.qmodel
            spectrum = qmodel.get_spectrum_vs_paramvals(param_name = "flux",  param_vals = flux)
            return (spectrum[1]-spectrum[0]) * 1e9 
        else:
            Ej_eff = self.Ej() * sqrt(cos(pi * flux) ** 2 + self.d**2 * sin(pi * flux) ** 2)
            return (sqrt(8 * Ej_eff * self.Ec())**0.5 - self.Ec()) # Fallback to approximate with the transmon formula


    def __str__(self):
        return super().__str__() + "\nFlux = \t%3.2f" % self.flux
