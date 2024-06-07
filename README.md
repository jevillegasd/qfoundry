# qfoundry
 Help functions for the calculation of superconductive circuits fro the quantum foundry at TII.

## Installation
The easier way to install the qfoundry module is using pip and cloning all the files directly from the repository. For this clone the files to the desired folder in your computer and run ``pip install .`` after having activated the desired environment. Since the module is actively being updated, using the editable option ``-e`` in pip will allow for changes in the module files without the need to reinstall it.
```bash
git clone https://github.com/jevillegasd/qfoundry
cd ~/qfoundry/src/qfoundry/
pip install -e .
```

## Usage

### Waveguides and Resonators
Coplanar waveguides (CPW) are the main builiding block of most circuits. A superconfuctive model of waveguides is directly implemented in the resonator.cpw module. A new CPW can be instantiated as 
```python
from qfoundry.resonator  import cpw
from IPython.display import display, Math

epsilon_r = 11.7            #Intrinsic Silicon
h = 525.      #Substrate Height in [μm]
w = 15        #cpw_width in [μm]
s = 7.5       #cpw_spacing in [μm]
t = 0.1       #cpw_thickness in [μm]

wg = cpw(epsilon_r,h,w,s,t)
display(Math(r'Z_0 = %2.2f\ \Omega,\ \epsilon_{eff} = %2.2f'%(wg.Z_0, wg.epsilon_ek)))
```
> $Z_0=49.22\Omega,\ \epsilon_{eff}=6.35$

Using the generated waveguide, an instance of a resonator can be cretaed using the ``cpw_resonator`` class. Resonatros can be defined using either the length of the resonator or the desired resonance frequency as

```python
from qfoundry.resonator  import cpw_resonator

f0 = 6.8*1e9
length_factor = 2 #4:Quarterwave, 2:Halfwave, 1:Fullwave
n = 1 #Mode number

res = cpw_resonator(wg, frequency = f0, length_f = length_factor, n=n)
```
Parameters like resonator capacitance, kinetic inductances and quality factors are directly calculated and are readily accesible. Additionaly, the module creates a simple RCL impedance model that can be used for simple classical calculations using impedance based operattions. These can be sued to plot the freqquemncy response of the resonator for example using
```python
import numpy as np 
frqs = np.linspace(6.6,7,1001)*1e9
S21 = res.Z(frqs)

f, axs = plt.subplots(1, 2, figsize=(12, 5))
plt.subplot(1,2,1)
fig = plt.plot(frqs,np.real(S21))
plt.subplot(1,2,2)
fig = plt.plot(frqs,np.imag(S21))
```

![image](https://github.com/jevillegasd/qfoundry/assets/14344419/8197201d-57d6-4959-8e76-0b02a94dcb08)

### Qubits


```python
```
