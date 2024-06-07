from qfoundry.resonator import cpw, circuit, cpw_resonator

import scqubits as scq

from scipy.constants import Boltzmann  as k_B
from scipy.constants import e  as e_0
from scipy.constants import Planck  as h_0
from numpy import sqrt, pi, tanh, abs


class sc_metal:
    '''
        Superconductive metal.
        Modelled only from its critical temperature.
    '''
    def __init__(self, Tc, T=20e-3):
        self.Tc = 1.14

    def sc_gap(self):
        if self.T< 0.1:
            return 1.764*k_B*self.Tc
        else:
            return 3.076*k_B*sqrt(1-T/self.Tc)
        
    def sc_gap_eV(self):
        return self.sc_gap()/e_0
    

class transmon:
    '''
    Single Jucntion Qubit
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
    '''
    def __init__(self,
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
                 ng =0.3, #Offset Charge
                 ncut = 40,
                 truncated_dim = 10
                 ):
        self.mat = mat
        self.T = T
        self.mat.T = T
        self.R_jx = R_jx

        if (R_j == 0.0) & (E_j == 0.0):
            Exception('Either E_j or R_j need to be specified.')
        elif R_j == 0.0:
            Ic = E_j*2*e_0*2*pi
            self.R_j = pi*self.mat.sc_gap()/(2*e_0*Ic)*tanh(self.mat.sc_gap()/(2*k_B*self.T)) - R_jx
        else:
            self.R_j = R_j     
            
        self.C_sum = C_sum
        self.C_g = C_g
        self.C_k = C_k
        self.C_xy = C_xy
        self.C_in = C_in
        self.Cr = res_ro.C
        self.res_ro = res_ro
        
        self.qmodel = scq.Transmon(  EJ=self.Ej()/1e9,
                                    EC=self.Ec()/1e9,
                                    ng=ng,
                                    ncut=ncut,
                                    truncated_dim=truncated_dim)
        
        self.alpha = self.qmodel.anharmonicity()*1e9 #-self.Ec()
        self.Delta = abs(self.res_ro.f0()-self.f01())
        if kappa == 0.0:
            self.kappa = self.res_ro.kappa_ext()
        else:
            self.kappa = kappa

    def Ic(self):
        self.mat.T = self.T
        return pi*self.mat.sc_gap()/(2*e_0*(self.R_j+self.R_jx))*tanh(self.mat.sc_gap()/(2*k_B*self.T))
    
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
        return e_0*self.C_g/(self.C_g+self.C_sum)*sqrt(2*self.res_ro.f0()/(h_0*self.res_ro.C))
    
    def chi(self):
        '''
        Dispersive shift
        '''
        return -(self.g01()**2)/(self.Delta)*(1/(1+self.Delta/self.alpha))

    def f01(self):
        '''
        Qubit 01 frequency
        '''
        return self.qmodel.E01()*1e9
        #return ((8*self.Ej()*self.Ec())**0.5-self.Ec())
    
    def f02(self):
        '''
        Qubit 02 frequency
        '''
        return (self.f01()*2+self.alpha)/2
    
    def SNR(self, T1 = 100e-6):
        '''
        SNR
        '''
        gamma_1 = 1/T1
        return self.kappa*self.chi()**2/(gamma_1*(self.kappa**2/4+self.chi()**2))
    
    def T1_max(self):
        '''
        Higher bound of T1
        '''
        return (self.Delta)**2/(self.g01()**2*self.kappa)
    
    def __str__(self):
            return ("Ec = \t%3.2f MHz \nEj = \t%3.2f GHz \nEJ/EC= \t%1.2f\nf_01 = \t%3.2f GHz \n" \
                "f_02 = \t%3.2f GHz \ng_01 = \t%3.2f MHz \nchi =\t%3.2f MHz \nT1_max =\t%3.2f us\n" \
                "alpha =\t%3.2f MHz"%(
                self.Ec()*1e-6, 
                self.Ej()*1e-9, 
                self.Ej()/self.Ec(),
                self.f01()*1e-9,
                self.f02()*1e-9,      
                self.g01()*1e-6,  
                self.chi()*1e-6,
                self.T1_max()*1e6,
                self.alpha*1e-6
                )
            )
                
                                                