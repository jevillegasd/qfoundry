# qfoundry
Utilities for the calculation, modeling, and simulation of superconducting microwave circuits for quantum devices.

Documentation: see docs/ or build with MkDocs (instructions below).

Versioning: follows PEP 440. See CHANGELOG.md for release notes.

## Installation
Install with pip from the repository root. Use editable mode for development.
```bash
git clone https://github.com/jevillegasd/qfoundry
cd qfoundry
pip install -e .
```

Optional: simulation extras (FEM-based capacitance tools):
```bash
pip install -e .[simulation]
```

## Usage

### Waveguides and resonators
Coplanar waveguides (CPW) are a core building block. A superconductive CPW model is implemented in `qfoundry.resonator.cpw`. Create a CPW:
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

Create a resonator from the CPW using either length or target frequency:

```python
from qfoundry.resonator  import cpw_resonator

f0 = 6.8*1e9
length_factor = 2 #4:Quarterwave, 2:Halfwave, 1:Fullwave
n = 1 #Mode number

res = cpw_resonator(wg, frequency = f0, length_f = length_factor, n=n)
```
Parameters like resonator capacitance, kinetic inductance, and Q are computed. A simple RLC model enables frequency-domain analysis:
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
Transmon and tunable transmon models are provided via `qfoundry.qubits` (wrapping scqubits for spectra and derived quantities). See `docs/qubits.md`.

## Docs
- Build and preview docs locally:
```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Versioning
- This project follows PEP 440. The current version is exposed as `qfoundry.__version__`.
- See CHANGELOG.md for releases.

## References
- See docs/references.md for key papers referenced in the implementation and formulas.
