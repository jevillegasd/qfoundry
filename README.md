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
Coplkanar waveguides (CPW) are the main builiding block of most circuits. A superconfuctive model of waveguides is directly implemented in the resonator.cpw module. A new CPW can be instantiated as 
```python
from qfoundry  import resonator.cpw

epsilon_r = 11.7            #Intrinsic Silicon
h = 525.      #Substrate Height in [μm]
w = 15        #cpw_width in [μm]
s = 7.5       #cpw_spacing in [μm]
t = 0.1       #cpw_thickness in [μm]


wg = cpw(epsilon_r,h,w,s, alpha = 0.8e-1)
display(Math(r'Z_0 = %2.2f\ \Omega,\ \epsilon_{eff} = %2.2f'%(wg.Z_0, wg.epsilon_ek)))

```
> 



```python
```


### Qubits


```python
```
