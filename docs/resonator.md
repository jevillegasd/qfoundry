# Waveguides and resonators

The CPW model and resonator utilities are in `qfoundry.resonator`.

Key equations reference:
- Ghione (1984): doi:10.1049/el:19840120
- Watanabe (1994): doi:10.1143/JJAP.33.5708
- Wallraff et al. (2008): arXiv:0807.4094

Example:
```python
from qfoundry.resonator import cpw, cpw_resonator

epsilon_r = 11.7
h, w, s, t = 525, 15, 7.5, 0.1
wg = cpw(epsilon_r, h, w, s, t)
res = cpw_resonator(wg, frequency=6.8e9, length_f=2)
res.f0(), res.Q()
```
