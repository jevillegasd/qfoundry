from scipy.constants import c, mu_0, epsilon_0
import numpy as np
from scipy import special as sp
import scqubits as scq

# def LCR_f(L: float,C:float,R:float=0.0) -> float:
#     return 1/np.sqrt(L*C)/2*np.pi

# def resonator_frequency(resonator_length:float, epsilon_e:float, length_factor: int=4) -> float:
#     return ((c)/((epsilon_e**.5)))*(1/(length_factor*resonator_length))

# def resonator_length(resonator_freq: float, epsilon_e: float, length_factor: int=4) -> float:
#     #quarter_wave resonator
#     return ((c)/((epsilon_e**.5)))*(1/(length_factor*resonator_freq))

class cpw:
    """ 
    A coplanar waveguide.

    Attributes
    ----------
        epsilon_r: float   
            Dielectric constant of the substrate
        height: float
            substrate\'s height in um
        width: float,    
            Microstrip width in um
        spacing: float
            Space from ground plane in um
        thickness: float = 0.1
            Superconductive metal layer thickness
        rho: float = 2.06e-3
            s
        tc: float= 1.23e-3
            s
        alpha: float = 2.4e-4):   
            attenuation cofficient m^-1 

    Methods
    -------

    """
    def __init__(self,    epsilon_r: float,       #Dielectric constant of the substrate
                        height: float,    #[length], substrate's height in um
                        width: float,    #[length], microstrip width in um
                        spacing: float,  #[length], Space from ground plane in um
                        thickness: float = 0.1,
                        rho: float = 2.06e-9, #normal state resisitivity of the thin film
                        tc: float= 1.23e-3,
                        alpha: float = 2.4e-2):   # attenuation cofficient m^-1 
        self.w = width #to match [1]
        self.s = spacing #to match [1]
        self.d = thickness #to match [1]
        self.h = height    
        self.rho_tc = rho
        self.tc = tc
        self.alpha = alpha # attenuation cofficient in m^-1 
        self.lambda_0 = 1.05e-3*np.sqrt(self.rho_tc/self.tc)    #Cohenrece Length

        self.L_m, self.L_k = self.inductances(self.w,self.s,self.d,self.h, rho, tc)
        self.C_m, self.epsilon_e = self.capacitances(self.w,self.s,self.h,epsilon_r)
        self.L = self.L_m+self.L_k

        self.Z_0 = np.sqrt(self.L_m/self.C_m)          #Equation (1) in [1]
        self.Z_0k = np.sqrt(self.L/self.C_m)
        self.epsilon_ek = c**2*(self.C_m*self.L)
        self.eta_0 = mu_0*c #~120*np.pi

    def LCR_f(L,C,R) -> float:
        return 1/np.sqrt(L*C)/(2*np.pi)

    def capacitances(self,w:float,s:float,h:float,eps_r:float):
        '''
        Calculate capacitances and effective permittivity of CPW
        '''
        k_0 = w/(w+2*s)
        k_1 = np.sinh(np.pi*w/(4*h))/np.sinh((np.pi*(w+2*s))/(4*h))
        k_0p = np.sqrt(1-np.square(k_0)) #k'_0 in book notation
        k_11 = np.sqrt(1-np.square(k_1)) #k'_1 in book notation

        K0  = sp.ellipk(k_0)
        K0p = sp.ellipk(k_0p)
        K1  = sp.ellipk(k_1)
        K1p = sp.ellipk(k_11)

        #Equation (2) in [1]
        eps_eff=1+(eps_r-1)*(K1*K0p)/(2*K1p*K0)
        C_m = 4*epsilon_0*eps_eff*K0/K0p
        return C_m, eps_eff
    
    def inductances(self,w:float,s:float,d:float,h:float, rho:float= 2.06e-3, tc:float= 1.23e-3):
        '''
        Calculate normal and kinetic inductances of a CPW
        '''
        k_0 = w/(w+2*s)
        k_1 = np.sinh(np.pi*w/(4*h))/np.sinh((np.pi*(w+2*s))/(4*h))
        k_0p = np.sqrt(1-np.square(k_0)) #k'_0 in book notation
        k_11 = np.sqrt(1-np.square(k_1)) #k'_1 in book notation

        K0  = sp.ellipk(k_0)
        K0p = sp.ellipk(k_0p)
        K1  = sp.ellipk(k_1)
        K1p = sp.ellipk(k_11)

        ######## Inductance per unit length #####################
        L_m= mu_0/4*K0p/K0  

        ######### Kinetic inductance ############################
        A = (1/(2*k_0**2*K0**2))
        g = A*(-    np.log(d/(4*w)) -   k_0*np.log(d/(4*(w+2*s)))   +   (2*(w+s)/(w+2*s))*np.log(s/(w+s)))

        #print g, a, b, c , d
        l0 = 1.05e-3*np.sqrt(rho/tc) # London penetration  depth
        L_k = mu_0*(l0**2)/(w*d)*g
        return L_m, L_k
    

class circuit():
    """ 
    A general RLC electrical circuit.

    Attributes
    ----------
        R:float=np.inf
            Circuit equivalent resistance in Ohms
        L:float = np.inf
            Circuit equivalent inductance in Henry
        C:float = 0.0
            Circuit equivalent capacitrance in Farads
        n:float = 0.0
            RCL resonance mode
        type: str = 'p'
            Circuit type. p is capacitor in parallel and s in series capacitor,

    Methods
    -------

    """

    def __init__(self, R:float=np.inf, L:float = np.inf, C:float = 0, n:float = 1, c_type: str = 'p'): #Type is p for parallel RLC and s for series
        self._R_ = R
        self._L_ = L
        self._C_ = C
        self.n = n
        self.c_type = c_type

    def _Zs_(self,f): 
        w = 2*np.pi*self._f0_()
        return self.R()  +  +1j*w*self.L()  +  1/(1j*w*self.C())
    
    def _Zp_(self,f,n): 
        w = 2*np.pi*self._f0_()
        return 1/(1/self.R()  +  1/(1j*w*self.L()*n)  +  1j*w*self.C()*n)

    def _f0_(self):
        return 1/(2*np.pi*np.sqrt(self.L()*self.C()))
    
    def Q(self):
        return self._R_*np.sqrt(self.C()/self.L()) 
    
    def Z(self,f):
        '''
            Frequency domain numeric transfer function (impedance)
        '''
        if self.c_type == 's':
            return  self._Zs_(f)
        else:
            return  self._Zp_(f)
        
    def __add__(self, o):
        return self.Z + o.Z 
    
    def __multiply__(self, o):
        return self.Z*o.Z 
    
    def R(self):
        return self._R_

    def C(self):
        return self._C_
    
    def L(self):
        return self._L_

class cpw_resonator(circuit):
    '''
    A coplanar waveguide resonator
    '''

    
    def __init__(self, wg: cpw, length:float = None, frequency: float = None, length_f:int = 2, n:int =1, Cg:float = 0.0, Ck:float = 0.0):
        self.wg = wg
        self.length_f = length_f #length factor: 4: quarter wavelength resonator
        self.n = n #mode number   
        self.Cin = Ck

        if frequency is None: # Input is length
            self.length = length
            self._C_ = self.wg.C_m*self.length + Cg + Ck
            self._L_ = self.wg.L_m*self.length/(self.n*2*np.pi)**2

        elif length is None: # Input is frequency
            self.length = self._get_length_(frequency*length_f, Cg + Ck, n = n) 
            self._L_ = self.wg.L_m*self.length/(self.n*2*np.pi)**2
            self._C_ = self.wg.C_m*self.length+ Cg + Ck
            
        else:
            return None
        self._R_ = wg.Z_0k/(self.wg.alpha*self.length*self.length_f)

        self.qmodel = scq.Oscillator(
            E_osc=self.f0()*1e-9,
            l_osc = self.length,
            truncated_dim=4  # up to 3 photons (0,1,2,3)
        )
    
    def _get_length_(self, f0, Cp:float = 0.0, n:int=1):
        from scipy.constants import c as c0
        def solve_quad(a,b,c):
            return  (-b + np.sqrt(b**2-4*a*c))/(2*a),  (-b - np.sqrt(b**2-4*a*c))/(2*a) 
        
        #If Cg + Ck == 0, the length is calculated using only the cpw
        if Cp >1e-20:
            Cm = self.wg.C_m
            Lm = self.wg.L_m
            w = 2*np.pi*f0*n
            Ls = Lm/(2*self.n*np.pi)**2

            l1,l2 = solve_quad(Cm*Ls*w**2, Ls*Cp*w**2, -1) 
            return max(l1,l2)
        else:
            return ((c0)/((self.wg.epsilon_ek**.5)))*(1/(f0*n))

    def Z_TL(self, f:np.array):
        fn = self.w0()/(2*np.pi)
        Z = self.wg.Z_0k/(self.wg.alpha*self.length + 1j*np.pi*(f-fn)/fn)
        return Z/Z.max()
    
    def Zp(self, f):
        return self._Zp_(f, self.length_f)

    def Z(self, f):
        '''
            frequency domain numeric transfer function (impedance), this overload the class:circuits impedance.
        '''
        return self.Z_TL(f) # Overload this function from the circuit class
    
    def w0(self):
        return 2*np.pi*self.f0()
    
    def f0(self):
        return self._f0_()/self.length_f #__f0__() calculates the fundamental LC resonance of the RCL circuit
    
    def kappa(self):
        return self.f0()/self.Q()
    
    def kappa_ext(self):
        return self.fwhm()
    
    def Q_ext(self, Cin=None):
        if Cin is None:
            Cin = self.Cin
        return np.pi/(4*(self.wg.Z_0*2*np.pi*self.f0()*Cin)**2)
        #return (1+(wr*C_k*R_L)**2)*(C+C_k))/(wr*C_k**2*R_L)  R_L=50 Ohm
    
    def fwhm(self, Cin = None):
        if Cin == None:
            Cin = self.Cin
        return self.f0()/self.Q_ext(Cin = Cin)
    
    def C(self):
        return self._C_

    def L(self):
        return self._L_

#
#    [1] Ghione 1984, doi: 10.1049/el:19840120
#    [2] Wanabe 1994, doi: 10.1143/JJAP.33.5708
#    [3] 
#