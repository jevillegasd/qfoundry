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

Transmon and flux-tunable transmon models are in `qfoundry.qubits`. Both expose a common `qubit` interface providing energies, ZPF amplitudes, and coupling parameters.

```python
from qfoundry.qubits import transmon

q = transmon(
    E_j = 15e9,    # Josephson energy  (Hz)
    E_c = 250e6,   # Charging energy   (Hz)
    C_g = 20e-15,  # Gate capacitance  (F)
)

print(f"f01  = {q.f01()*1e-9:.3f} GHz")
print(f"Ec   = {q.Ec()*1e-6:.1f} MHz")
print(f"Ej   = {q.Ej()*1e-9:.2f} GHz")
print(f"Ej/Ec= {q.Ej()/q.Ec():.1f}")
print(f"α    = {q.alpha()*1e-6:.1f} MHz")
print(f"g₀₁  = {q.g01()*1e-6:.2f} MHz  (to readout resonator)")
print(f"χ    = {q.chi()*1e-6:.3f} MHz  (dispersive shift)")
```

Key derived quantities accessible on any `transmon` / `tunable_transmon`:

| Method | Description |
|--------|-------------|
| `f01()`, `omega01()` | Qubit transition frequency (Hz / rad s⁻¹) |
| `Ej()`, `Ec()` | Josephson and charging energies (Hz) |
| `C()`, `L()` | Total capacitance (F) and Josephson inductance (H) |
| `alpha()` | Anharmonicity (Hz, negative for transmon) |
| `I_zpf()`, `V_zpf()` | Zero-point current / voltage fluctuations |
| `n_zpf()`, `phi_zpf()` | Charge and phase ZPF amplitudes |
| `g()`, `chi()` | Qubit–resonator coupling and dispersive shift (Hz) |
| `T1_max()` | Purcell-limited coherence time upper bound |

A flux-tunable transmon (SQUID-based) takes an additional `flux` parameter and `d` (SQUID asymmetry):

```python
from qfoundry.qubits import tunable_transmon

qt = tunable_transmon(
    E_j = 20e9,
    E_c = 220e6,
    flux = 0.0,    # Φ/Φ₀
    d    = 0.05,   # junction asymmetry
)
print(f"f01 at Φ=0:   {qt.f01(0.0)*1e-9:.3f} GHz")
print(f"f01 at Φ=0.5: {qt.f01(0.5)*1e-9:.3f} GHz")
```

### Qubit–qubit edges (couplers)

The `qfoundry.edges` module models directional couplings between qubits.
`edge(q0, q1)` designates `q0` as the *control* and `q1` as the *target*,
which matters for the asymmetric cross-resonance coefficients `nu` (IX) and `mu` (ZX).

Five coupler types are available:

| Class | Coupling mechanism |
|-------|--------------------|
| `capacitive_coupler` | Direct shunt capacitance $C_{12}$ |
| `inductive_coupler` | Mutual inductance $M$ |
| `bus_resonator_coupler` | Resonator-mediated exchange (dispersive regime) |
| `tunable_coupler` | Flux-tunable SQUID-based coupler (Chen 2014 / Yan 2018) |
| `hybrid_coupler` | Capacitive + inductive (3-D integrated / flip-chip) |

```python
from qfoundry.edges import capacitive_coupler, bus_resonator_coupler

# Direct capacitive coupling
edge_cap = capacitive_coupler(q0, q1, C_12=3e-15)  # 3 fF
print(f"g    = {edge_cap.g()*1e-6:.2f} MHz")
print(f"ζ    = {edge_cap.zeta()*1e-6:.3f} MHz  (ZZ)")
print(f"ν    = {edge_cap.nu():.3e}  (IX / rad·s⁻¹)")
print(f"μ    = {edge_cap.mu():.3e}  (ZX / rad·s⁻¹)")

# Bus-resonator mediated coupling
from qfoundry.resonator import cpw_resonator
from qfoundry.waveguides import cpw

bus = cpw_resonator(cpw(11.7, 0.1, 15, 7.5), frequency=7e9, length_f=2)
edge_bus = bus_resonator_coupler(q0, q1, bus, C_0r=5e-15, C_1r=5e-15)
print(f"g_eff = {edge_bus.g()*1e-6:.2f} MHz")
```

Every edge also exposes `hilbert_space()` to build a scqubits `HilbertSpace`
for full numerical diagonalisation:

```python
hs = edge_cap.hilbert_space()
hs.generate_lookup()
```

For the flux-tunable coupler, the zero-coupling flux point can be found automatically:

```python
from qfoundry.edges import tunable_coupler

tc = tunable_coupler(q0, q1, E_j_max=30e9, E_c=800e6, C_0c=5e-15, C_1c=5e-15)
phi_off = tc.flux_for_zero_coupling()
print(f"Zero-coupling at Φ/Φ₀ = {phi_off:.4f}")
```

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
