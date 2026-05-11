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
from scipy.constants import Planck as h_0, hbar, pi

from numpy import cos, sin, sqrt, tanh, abs, exp, pi 
from numpy import diag, ones, arange, diff, ndarray
from math import factorial

from scipy.linalg import eigh
import inspect

class transmon(circuit):
    """
    Single Junction Qubit
        E_j:    float               # Josephson energy
        C_sum:  float=67.5e-15,
        C_g:    float  =21.7e-15,
        C_k:    float  =36.7e-15,
        C_d:   float =0.0e-15,
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
        C_d: float = 0.0e-15, # Capacitance to ground
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
        self.C_d = C_d
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
        cls._Rx_ = kwargs.get("R_jx", 0.0)
        cls._Rj_ = Ic_to_R(i_c, mat=kwargs.get("mat", sc_metal(1.14, 25e-3))) - cls._Rx_
        kwargs["R_j"] = cls._Rj_
        kwargs["R_jx"] = cls._Rx_
        return cls(E_j=E_j, **kwargs)

    @classmethod
    def from_rj(cls, R_j: float, **kwargs):
        """
        Initialize transmon from junction resistance R_j.
        The saved junction resistance is R_j, but the effective resistance 
        used to calculate the qubit properties is R_j - R_jx.
        """
        from numpy import isnan
        cls._Rx_ = kwargs.get("R_jx", 0.0)
        if cls._Rx_ is None or isnan(cls._Rx_):
            cls._Rx_ = 0.0
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

        print(f"Calculated Ej: {ej*1e-9:.3f} GHz, Ec: {ec*1e-6:.3f} MHz, Ic: {ic*1e9:.3f} nA, Rj: {cls._Rj_+cls._Rx_:.3f} Ohm")
        kwargs["R_j"] = cls._Rj_
        kwargs["R_jx"] = cls._Rx_
        return cls(E_j=ej, E_c=ec, **kwargs)

    def alpha(self):
        """
        Anharmonicity
        """
        from numpy.linalg import LinAlgError 
        alpha = -self.Ec() # Analytical approximation for anharmonicity, in Hz
        if self.qmodel is not None:
            try: 
                return self.qmodel.anharmonicity() * 1e9
            except LinAlgError as e:
                return alpha
        return alpha

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
        n_j_annalytical = sqrt((1+j)/2)*(self.Ej() / (8*self.Ec()))**(1/4)
        return n_j_annalytical
    
    def g_j(self,j):
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
        return self.g()
    
    def g(self):
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

        Another option is to use the numerical charge matrix element n_matrix()[0,1]
        g = C_g*(2*Ec/e) * V_zpf * n_matrix()[0,1] / hbar

        """

        C_sum = self.C()
        f_r     = self.res_ro.f0() # Resonator angular frequency
        C_r     = self.res_ro.C() # Resonator capacitance
        V_zpf    = sqrt(h_0 * f_r / (2 * C_r)) # Zero point fluctuation voltage
        C_g   = self.C_g
        try:
            assert self.n01() is not None
        except:
            return C_g/2*sqrt(f_r*self.f01()/ (C_sum*C_r))  # Fallback if n01 is not available
        
        beta    = self.C_g / C_sum # Participation ratio
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

    def E_m(self, m):
        """
        Energy of the m-th level
        """
        from numpy.linalg import LinAlgError
        if self.qmodel is not None: 
            try:
                spectrum = self.qmodel.get_spectrum_vs_paramvals(param_name = "ng",  param_vals = [0, 0.5])
                E_m = spectrum.energy_table[m,0] # Energy at ng=0
                return E_m * 1e9
            except LinAlgError as e:
                return -self.Ej() + sqrt(8*self.Ej()*self.Ec())*(m+0.5) - self.Ec()*(6*m**2 + 6*m + 3)/12
            
        else:
            return -self.Ej() + sqrt(8*self.Ej()*self.Ec())*(m+0.5) - self.Ec()*(6*m**2 + 6*m + 3)/12
    

    def E01(self):
        from numpy.linalg import LinAlgError
        if self.qmodel is not None: 
            try:
                return self.qmodel.E01() * 1e9
            except LinAlgError as e:
                return sqrt(8*self.Ej()*self.Ec())*(0.5) - self.Ec()
        else:
            return self.Ej() + sqrt(8*self.Ej()*self.Ec())*(0.5) - self.Ec()


    def f01(self):
        """
        Qubit 01 frequency, in Hz (equivalent to E01)
        """
        return self.E01()
       

    def omega01(self):
        """
        Qubit 01 angular frequency
        """
        return 2 * pi * self.f01()

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
        Higher bound of T1 (Purcell limit)
        https://arxiv.org/pdf/cond-mat/0703002 eq 4.7
        """
        return (self.delta() / self.g01()) ** 2 / (self.res_ro.kappa_ext())
    
    def T1_drive(self, Z0=50):
        """
        T1 limited by the drive coupling
        """
        from numpy import floating, ndarray
        C_c = self.C_d

        # Check if C_c is a positive float or a list/array of two elements
        if not (isinstance(C_c, (float, floating)) and C_c > 0) and not (isinstance(C_c, (list, ndarray)) and len(C_c) == 2):
            print("C_d should be a positive float or a list/array of two elements.")
            return float('inf')
        else:
            if isinstance(C_c, (list, ndarray)):
                assert len(C_c) == 2, "C_d must be a list/array of two elements."
                return self.C() / (Z0 * (self.omega01())**2 * (diff(C_c)[0]**2)) / (2*pi) # * 4 # Double check this derivation
            return self.C() / (Z0 * (self.omega01())**2 * (C_c)**2) / (2*pi)

    def omega_rabi(self, V_rms):
        """
        Rabi frequency under a drive with root mean square voltage V_rms
        """
        from numpy import floating
        C_sum = self.C()
        C_c = self.C_d if isinstance(self.C_d, (floating, np.floating)) else abs(diff(self.C_d)[0])

        try:
            n01 = self.n01()
        except:
            n01 = 1.2

        beta = C_c / C_sum  # Participation ratio
        return (2 * e_0 * n01 * beta * V_rms)/hbar # Rabi frequency
    
    def tau_rabi(self, P_in=-63, Z0=50):
        """
        Rabi time under a drive with root mean square voltage V_rms
        Arguments:
        P_in : float
            Input power in Watts (if negative, assumed to be in dBm)
        Z0 : float
            Characteristic impedance of the drive line (default: 50 Ohm)

        """
        if P_in <= 0:
            # Power is probably in dB
            P_in = 10 ** (P_in / 10) * 1e-3  # Convert dBm to Watts

        V_rms = sqrt(2 * Z0 * P_in)  # Root mean square voltage
        tau_pi = pi / self.omega_rabi(V_rms, Z0)  # Time for a pi pulse
        return tau_pi
    
    def epsilon_m(self, m = 0):
        """
        Charge dispersion for the m-th level
        https://arxiv.org/pdf/cond-mat/0703002 eq 3.5
        """
        from numpy.linalg import LinAlgError

        eps_m = (-1)**m * self.Ec() * (2**(4*m+5) / factorial(m)) * sqrt(2/pi) * (self.Ej()/(8*self.Ec()))**(m/2 + 3/4) * exp(-sqrt(8*self.Ej()/self.Ec())) # Analytical approximation for charge dispersion, valid in the transmon regime (EJ/EC >> 1)
        if self.qmodel is not None:
            try:
                spectrum = self.qmodel.get_spectrum_vs_paramvals(param_name = "ng",  param_vals = [0, 0.5])
                epsilon_m = abs(spectrum.energy_table[m,1] - spectrum.energy_table[m,0])
                return epsilon_m * 1e9
            except LinAlgError as e:
                return eps_m
        else:
            return eps_m


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
    flux = 0
    def __init__(self, flux=0.0, d=1, *args, **kwargs):
        self.flux = flux
        self.d = d  # Assymetry parameter
        self.ng = kwargs.get("ng", 0.3)  # Offset charge
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
        E_j = i_c / (4 * e_0 * pi)
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
            flux = [self.flux]
        elif isinstance(flux, (int, float)):
            flux = [flux]
        if self.qmodel is not None:
            # By default return the f01 using scqubits
            spectrum = self.qmodel.get_spectrum_vs_paramvals(param_name = "flux",  param_vals = flux)
            f01 = spectrum.energy_table[:,1]-spectrum.energy_table[:,0]
            return f01 * 1e9 if len(f01) > 1 else f01[0] * 1e9
        else:
            Ej_eff = self.Ej() * sqrt(cos(pi * flux) ** 2 + self.d**2 * sin(pi * flux) ** 2)
            return (sqrt(8 * Ej_eff * self.Ec())**0.5 - self.Ec()) # Fallback to approximate with the transmon formula


    def __str__(self):
        return super().__str__() + "\nFlux = \t%3.2f" % self.flux
