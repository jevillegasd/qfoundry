# Qubits

Transmon and tunable transmon models are provided in `qfoundry.qubits`. These models wrap `scqubits` objects to provide easy access to energy spectra and derived parameters. The underlying numerical analysis and parameter extraction are based on the framework provided by `scqubits`, as detailed in the paper by Groszkowski & Koch (2021), with analysis techniques guided by resources like Chitta et al. (2022).

References:
- Koch et al., PRA 76, 042319 (2007)
- Krantz et al., Applied Physics Reviews 6, 021318 (2019) arXiv:1904.06560
- Groszkowski, P. & Koch, J. (2021). "scqubits: a Python package for superconducting qubits." *Quantum* 5, 463.
- Chitta, A., et al. (2022). "Numerical analysis of superconducting qubits."

Example:
```python
from qfoundry.qubits import transmon
from qfoundry.resonator import cpw, cpw_resonator

wg = cpw(11.7, 525, 15, 7.5, 0.1)
res = cpw_resonator(wg, frequency=6.8e9, length_f=2)
q = transmon(R_j=8e3, C_sum=70e-15, C_g=20e-15, C_k=35e-15, res_ro=res)
q.f01(), q.alpha(), q.chi()
```
