
from scipy.constants import elementary_charge as e_0
from scipy.constants import h, hbar
from numpy import sqrt, pi
from qfoundry.resonator import circuit

def Cs_to_E(C):
    return e_0**2/(2*C)/h

def Cg_to_E(Cg, C1, C2):
    '''
    Using Vi = (2e/Ci)ni and the Hamiltonian H_int = Hint = CgV1V2
    So that in the perturbative regime, H_int = g_int*n1*n2
    H_int = 4*e_0**2*Cg/(C1*C2)/h
    https://arxiv.org/pdf/1904.06560 (Krantz 2021) eq. 27
    '''
    return 4*e_0**2*Cg/(C1*C2-Cg**2)/h

def cap_coupling(Ck,C1,C2,w1,w2):
    '''
    The standard formula for capacitive coupling between harmonic modes [1].
    The entries w1 and w2 in radian frequencies.
    [1] E. Jeffrey, Phys. Rev. Lett. 112, 190504, https://arxiv.org/pdf/1401.0257
    '''
    return 0.5*Ck/(sqrt(C1*C2))*sqrt(w1*w2)

def C_to_g(Cg, C_sum, f0,Cr):
    # Wallraff et al. 2004 
    return e_0*Cg/(Cg+C_sum)*sqrt(2*2*pi*f0/(hbar*Cr))/(2*pi)

def g_hm(Cg, hm0:circuit, hm1:circuit):
    '''
    Capacitive coupling between harmonic circuits.
    '''
    #return Cg_to_E(Cg, hm0.C(), hm1.C())
    return cap_coupling(Cg,hm0.C(),hm1.C(),hm0.f0(),hm1.f0())



