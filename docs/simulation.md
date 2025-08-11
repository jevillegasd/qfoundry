# Simulation

Capacitance simulations use femwell + scikit-fem. See `qfoundry.simulation.capacitance`.

```python
from qfoundry.simulation.capacitance import coplanar_capacitor

sim = coplanar_capacitor(width=20, spacing=10, thickness=0.2, substrate_heigth=525)
C_per_length = sim.capacitance()  # F/m
```

Note: These solvers require additional system dependencies and can be slow for fine meshes.

Install extras if imports fail:
```bash
pip install -e .[simulation]
```
