from qfoundry.resonator import cpw, circuit, cpw_resonator
from qfoundry.utils import C_to_g

import scqubits as scq

from scipy.constants import Boltzmann  as k_B
from scipy.constants import e  as e_0
from scipy.constants import Planck  as h_0
from scipy.constants import fine_structure as alpha
from numpy import sqrt, cos, pi, tanh, abs

Zvac = 376.730313461 # Impedance of free space
Rk = h_0/(e_0**2)   # Resistance quantum (von Klitzing constant)
# alpha = 0.0072973525643 # Fine structure constant (Zvac/(2*Rk))

class __sc_metal__:
    '''
        Superconductive metal.
        Modelled only from its critical temperature.
    '''
    def __init__(self, Tc, T=20e-3):
        self.Tc = 1.14 if Tc is None else Tc
        self.T = T

    def sc_gap(self):
        if self.T< 0.1:
            return 1.764*k_B*self.Tc
        else:
            return 3.076*k_B*sqrt(1-self.T/self.Tc)
        
    def sc_gap_eV(self):
        return 2*self.sc_gap()/e_0
    

class transmon(circuit):
    '''
    Single Junction Qubit
        R_j:float=0.0,       # Total junction resistance
        E_j:float=0.0,
        C_sum:float=67.5e-15,
        C_g:float  =21.7e-15,
        res_ro     = cpw_resonator(cpw(11.7,0.1,12,6, alpha=2.4e-2),frequency = 7e9, length_f = 2),    #Readout Resonator
        R_jx:float = 0.0,       # Resistance correction factor
        mat = sc_metal(1.14),
        T = 20.e-3,
        kappa = 0.0,
        ng =0.3 #Offset Charge
    NOTE: Energies are in E/h (not E/hbar)
    '''

    #get
    def qmodel(self):
        return self.__qmodel__
    
    def mat(self):
        return self.__mat__
    
    @property
    def fres(self):
        '''Resonator frequency'''
        return self.__res_ro__.f0()
    
    @property
    def f01(self):
        '''Qubit frequency'''
        return self._f01_()
    
    @property
    def f02(self):
        '''Qubit 02 frequency'''
        return self._f02_()

    @property
    def delta(self):
         ''' Detuning'''
         return abs(self.fres-self.f01)

    @property
    def alpha(self):
        '''Qubit anharmonicity'''
        return self.__qmodel__.anharmonicity()*1e9 #-self.Ec()
    
    @property
    def kappa(self):
        '''Readout resonator linewidth'''
        return self.__kappa__
    
    @property
    def g(self):
        '''Coupling strength'''
        return self.g01()
    
    @property
    def chi(self):
        '''Lamb Dispersive shift'''
        return self.__chi__()
    
    @property
    def capacitive_energy(self):
        '''Capacitive energy'''
        return self.Ec()
    
    @property
    def josephson_energy(self):
        '''Josephson energy'''
        return self.Ej()
    
    @property
    def energy_ratio(self):
        '''Ej/Ec'''
        return self.Ej()/self.Ec()
    
    
    

    @property
    def T1_max(self):
        '''Purcell limited T1'''
        return (self.delta)**2/(self.g01()**2*self.kappa)

    def __init__(self,
                 R_j:float=None,       # Total junction resistance
                 E_j:float=None,
                 C_sum:float=67.5e-15,
                 C_g:float  =21.7e-15,
                 res_ro     = cpw_resonator(cpw(11.7,0.1,12,6, alpha=2.4e-2),frequency = 7e9, length_f = 2),    #Readout Resonator
                 R_jx:float = 0.0,       
                 mat = __sc_metal__(Tc=1.14),
                 T = 20.e-3,
                 kappa = None,
                 ng =0.3, #Offset Charge
                 ncut = 40,
                 truncated_dim = 10
                 ):
        self.__mat__ = mat
        self.__mat__.T = T
        self.T = T

        self.C_sum =    C_sum   # Total capacitance
        self.C_g =      C_g     # Coupling capacitance (to responator)
        self.R_jx =     R_jx    # Junction resistance correction factor

        self.__res_ro__ = res_ro
        self.__kappa__ = res_ro if kappa is None else kappa

        if (R_j is None) & (E_j is None):
            raise(Exception('Either E_j or R_j need to be specified.'))
        elif R_j is None:
            Ic = E_j*2*e_0*2*pi
            self.R_j = pi*self.__mat__.sc_gap()/(2*e_0*Ic)*tanh(self.__mat__.sc_gap()/(2*k_B*self.T)) - R_jx
        else:
            self.R_j = R_j     
        
        self.__qmodel__ = scq.Transmon( EJ=self.Ej()/1e9,
                                    EC=self.Ec()/1e9,
                                    ng=ng,
                                    ncut=ncut,
                                    truncated_dim=truncated_dim)

    def L(self, phi = 0.):
        '''
        RLC circuit modcel josephson inductance for the ground state
        '''
        return self.Lj(phi)

    def Ic(self):
        '''
        Critical current
        Ref: V. Ambegaokar and A. Baratoff, “Tunneling between superconductors,” Phys. Rev. Lett., Vol. 11, p. 104, 15 July 1963 (Erratum of Phys. Rev. Lett., Vol. 10, pp. 486–491, 1 June 1963).
        https://doi.org/10.1103/PhysRevLett.10.486
        '''
        T = self.__mat__.T
        sc_gap = self.__mat__.sc_gap()
        return pi*sc_gap/(2*e_0*(self.R_j+self.R_jx))*tanh(sc_gap/(2*k_B*T))
    
    def Lj(self, phi=0):
        '''
        Josephson inductance
        '''
        phi_0 = h_0/(2*e_0)
        return phi_0 / (2*pi*self.Ic()*cos(phi))
    


    def fj(self, V = None):
        '''
        Josephson frequency
        
        '''
        V = self.__mat__.sc_gap_eV() if V is None else V
        return 2*e_0*V/h_0

    def Ec(self):
        '''
        Capacitive energy 
        '''
        return e_0**2/(2*self.C_sum)/h_0
    
    def Ej(self):
        '''
        Josephson energy
        '''
        return self.Ic()/(2*e_0)/(2*pi)

    def g01(self):
        '''
        Qubit and resonator coupling strength
       
        '''
        f0 = self.__res_ro__.f0()
        omega = 2*pi*f0
        Zr = 50 #self.__res_ro__.impedance()
        Cr = self.__res_ro__.C()
        Cg = self.C_g
        C_sum = self.C_sum

        return omega*Cg/(C_sum)*(self.Ej()/(2*self.Ec()))**(1/4)*sqrt(pi*Zr/Zvac)*sqrt(2*pi*alpha)/(2*pi) # https://arxiv.org/pdf/2005.12667 eq. 33
        #return C_to_g(Cg, C_sum, f0,Cr)
    
    def __chi__(self):
        '''
        Lamb Dispersive shift (~Stark Shift /2)
        '''
        return -((self.g01()**2/self.delta)) #* 1/(1+self.Delta/self.alpha)

    def _f01_(self):
        '''
        Qubit 01 frequency
        '''
        return self.__qmodel__.E01()*1e9
        #return ((8*self.Ej()*self.Ec())**0.5-self.Ec())
    
    def _f02_(self):
        '''
        Qubit 02 frequency /2
        '''
        return (self.f01*2+self.alpha)/2
    
    def f0(self):
        '''
        Wrapper for qubit frequency for RLC circuit model
        '''
        return self.f01
    
    def SNR(self, T1 = 100e-6):
        '''
        SNR
        '''
        gamma_1 = 1/T1
        return self.kappa*self.chi**2/(gamma_1*(self.kappa**2/4+self.chi**2))
    
    def C(self):
        return self.C_sum

    
    def __str__(self):
            return ("Ec = \t%3.2f MHz \nEj = \t%3.2f GHz \nEJ/EC= \t%1.2f\nf_01 = \t%3.2f GHz \n" \
                "f_02/2 = \t%3.2f GHz \ng_01 = \t%3.2f MHz \nchi =\t%3.2f MHz \nT1_max =\t%3.2f us\n" \
                "alpha =\t%3.2f MHz\nL_jj =\t%3.2f nH\n"%(
                self.Ec()*1e-6, 
                self.Ej()*1e-9, 
                self.Ej()/self.Ec(),
                self.f01*1e-9,
                self.f02*1e-9,      
                self.g01()*1e-6,  
                self.chi*1e-6,
                self.T1_max*1e6,
                self.alpha*1e-6,
                self.Lj(0)*1e9
                )
            )
    
    def properties(self):
        class_items = self.__class__.__dict__.items()
        return dict((k, getattr(self, k)) 
                    for k, v in class_items 
                    if isinstance(v, property))
    
   

                
                                                