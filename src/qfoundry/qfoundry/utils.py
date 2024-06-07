
from scipy.constants import elementary_charge as e
from scipy.constants import h

def Cs_to_E(C):
    return e**2/(2*C)/h

def Cg_to_E(Cg, C1, C2):
    return 4*e**2*Cg/(C1*C2-Cg**2)/h
