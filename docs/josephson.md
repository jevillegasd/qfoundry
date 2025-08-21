# Josephson Junctions

Analysis and characterization tools for Josephson junctions, including room temperature I-V measurements.

The `qfoundry.josephson` module provides tools for fitting experimental I-V data to various tunneling models:

- **Direct tunneling** (low voltage regime)
- **Trap-assisted tunneling** (intermediate voltage)  
- **Fowler-Nordheim tunneling** (high voltage regime)

Example:
```python
from qfoundry.josephson import analyze_junction_iv, JosephsonJunctionAnalyzer

# Quick analysis
results = analyze_junction_iv(V_data, I_data, model='composite')

# Detailed analysis
analyzer = JosephsonJunctionAnalyzer()
popt, pcov = analyzer.fit_iv_data(V_data, I_data, model='composite')
analyzer.plot_fit_results(V_data, I_data, popt)
```

References:
- Fowler, R. H. & Nordheim, L. (1928). "Electron emission in intense electric fields." Proc. R. Soc. Lond. A 119, 173-181.
- Simmons, J. G. (1963). "Generalized formula for the electric tunnel effect." J. Appl. Phys. 34, 1793-1803.
