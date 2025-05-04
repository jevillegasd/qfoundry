
from scipy.constants import elementary_charge as e
from scipy.constants import h
from numpy import sqrt
from qfoundry.resonator import circuit


def Cs_to_E(C):
    return e**2/(2*C)/h

def Cg_to_E(Cg, C1, C2):
    return 4*e**2*Cg/(C1*C2-Cg**2)/h

def C_to_g(Cg, C_sum, f0,Cr):
    
    return e*Cg/(Cg+C_sum)*sqrt(2*f0/(h*Cr))

def g_hm(Cg, hm0:circuit, hm1:circuit):
    '''
    Capacitive coupling between harmonic circuits.
    '''
    return coupling(Cg,hm0.C(),hm1.C(),hm0.f0(),hm1.f0())

def coupling(Ck,C1,C2,w1,w2):
    '''
    The standard formula for capacitive coupling between harmonic modes [1].
    The entries w1 and w2 radian frequencies.
    [1] E. Jeffrey, Phys. Rev. Lett. 112, 190504, https://arxiv.org/pdf/1401.0257
    '''
    return 0.5*Ck/(sqrt(C1*C2))*sqrt(w1*w2)


 # 