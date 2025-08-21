"""Josephson junction analysis and characterization.

This module provides tools for analyzing room temperature I-V measurements
of Josephson junctions using various tunneling models.

References
----------
- Fowler, R. H. & Nordheim, L. (1928). "Electron emission in intense electric fields."
  Proc. R. Soc. Lond. A 119, 173-181.
- Simmons, J. G. (1963). "Generalized formula for the electric tunnel effect between
  similar electrodes separated by a thin insulating film." J. Appl. Phys. 34, 1793-1803.
- Lang et al, Wafer-Scale Characterization of Al/AlxOy/Al Josephson Junctions at Room Temperature https://arxiv.org/pdf/2504.16686
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class JosephsonJunctionAnalyzer:
    """
    Analyzer for room temperature I-V characteristics of Josephson junctions.

    Provides fitting methods for different tunneling regimes:
    - Direct tunneling (low voltage)
    - Trap-assisted tunneling (intermediate voltage)
    - Fowler-Nordheim tunneling (high voltage)
    """

    def __init__(self):
        # Physical constants
        self.hbar = 1.055e-34  # J·s
        self.m_e = 9.11e-31  # kg (electron mass)
        self.e = 1.602e-19  # C (elementary charge)
        self.kB = 1.381e-23  # J/K (Boltzmann constant)
        self.B_FN = 6.83e9  # V·m⁻¹·eV⁻³⁄² (Fowler-Nordheim constant)

    def direct_tunneling(self, V, A, d, phi):
        """
        Direct tunneling model for low voltage regime.

        Parameters
        ----------
        V : array_like
            Applied voltage (V)
        A : float
            Pre-factor amplitude
        d : float
            Barrier thickness (m)
        phi : float
            Barrier height (eV)

        Returns
        -------
        array_like
            Current (A)
        """
        pre_factor = A
        exponent = (2 * d * np.sqrt(2 * self.m_e * phi * self.e)) / self.hbar
        return pre_factor * V * np.exp(-exponent)

    def trap_assisted_tunneling(self, V, A, Nt, Et, T):
        """
        Trap-assisted tunneling model for intermediate voltage regime.

        Parameters
        ----------
        V : array_like
            Applied voltage (V)
        A : float
            Pre-factor amplitude
        Nt : float
            Trap density (m⁻³)
        Et : float
            Trap energy (eV)
        T : float
            Temperature (K)

        Returns
        -------
        array_like
            Current (A)
        """
        return A * V * Nt * np.exp(-Et / (self.kB * T))

    def fowler_nordheim(self, V, A, phi, d):
        """
        Fowler-Nordheim tunneling model for high voltage regime.

        Parameters
        ----------
        V : array_like
            Applied voltage (V)
        A : float
            Pre-factor amplitude
        phi : float
            Barrier height (eV)
        d : float
            Barrier thickness (m)

        Returns
        -------
        array_like
            Current (A)
        """
        return A * V**2 * np.exp(-self.B_FN * phi**1.5 / (V * d))

    def composite_model(self, V, A1, d1, phi1, A2, Nt2, Et2, T2, A3, phi3, d3):
        """
        Composite model combining all three tunneling regimes.

        Parameters
        ----------
        V : array_like
            Applied voltage (V)
        A1, d1, phi1 : float
            Direct tunneling parameters
        A2, Nt2, Et2, T2 : float
            Trap-assisted tunneling parameters
        A3, phi3, d3 : float
            Fowler-Nordheim tunneling parameters

        Returns
        -------
        array_like
            Total current (A)
        """
        I1 = self.direct_tunneling(V, A1, d1, phi1)
        I2 = self.trap_assisted_tunneling(V, A2, Nt2, Et2, T2)
        I3 = self.fowler_nordheim(V, A3, phi3, d3)
        return I1 + I2 + I3

    def fit_iv_data(self, V_data, I_data, model="composite", initial_guess=None):
        """
        Fit I-V data to specified tunneling model.

        Parameters
        ----------
        V_data : array_like
            Voltage data (V)
        I_data : array_like
            Current data (A)
        model : str, default 'composite'
            Model to fit: 'direct', 'trap_assisted', 'fowler_nordheim', or 'composite'
        initial_guess : list, optional
            Initial parameter guess for fitting

        Returns
        -------
        tuple
            Fitted parameters and covariance matrix
        """
        if model == "composite":
            if initial_guess is None:
                initial_guess = [1e-6, 1e-9, 0.5, 1e-6, 1e18, 0.3, 300, 1e-6, 0.5, 1e-9]
            return curve_fit(
                self.composite_model, V_data, I_data, p0=initial_guess, maxfev=10000
            )
        elif model == "direct":
            if initial_guess is None:
                initial_guess = [1e-6, 1e-9, 0.5]
            return curve_fit(
                self.direct_tunneling, V_data, I_data, p0=initial_guess, maxfev=10000
            )
        elif model == "trap_assisted":
            if initial_guess is None:
                initial_guess = [1e-6, 1e18, 0.3, 300]
            return curve_fit(
                self.trap_assisted_tunneling,
                V_data,
                I_data,
                p0=initial_guess,
                maxfev=10000,
            )
        elif model == "fowler_nordheim":
            if initial_guess is None:
                initial_guess = [1e-6, 0.5, 1e-9]
            return curve_fit(
                self.fowler_nordheim, V_data, I_data, p0=initial_guess, maxfev=10000
            )
        else:
            raise ValueError(f"Unknown model: {model}")

    def plot_fit_results(
        self,
        V_data,
        I_data,
        fitted_params,
        model="composite",
        title="I-V Characteristic Fit",
    ):
        """
        Plot experimental data and fitted model.

        Parameters
        ----------
        V_data : array_like
            Voltage data (V)
        I_data : array_like
            Current data (A)
        fitted_params : array_like
            Fitted model parameters
        model : str, default 'composite'
            Model type used for fitting
        title : str
            Plot title
        """
        plt.figure(figsize=(8, 5))
        plt.plot(V_data, I_data, "b.", label="Experimental Data", markersize=4)

        if model == "composite":
            I_fit = self.composite_model(V_data, *fitted_params)
        elif model == "direct":
            I_fit = self.direct_tunneling(V_data, *fitted_params)
        elif model == "trap_assisted":
            I_fit = self.trap_assisted_tunneling(V_data, *fitted_params)
        elif model == "fowler_nordheim":
            I_fit = self.fowler_nordheim(V_data, *fitted_params)

        plt.plot(
            V_data, I_fit, "r-", label=f"Fitted {model.title()} Model", linewidth=2
        )
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def generate_synthetic_data(
        self, V_range=(0.01, 1.5), num_points=100, noise_level=1e-8, params=None
    ):
        """
        Generate synthetic I-V data for testing.

        Parameters
        ----------
        V_range : tuple
            Voltage range (V_min, V_max)
        num_points : int
            Number of data points
        noise_level : float
            Gaussian noise standard deviation
        params : list, optional
            Model parameters [A1, d1, phi1, A2, Nt2, Et2, T2, A3, phi3, d3]

        Returns
        -------
        tuple
            (V_data, I_data) arrays
        """
        if params is None:
            params = [1e-6, 1e-9, 0.5, 1e-6, 1e18, 0.3, 300, 1e-6, 0.5, 1e-9]

        V_data = np.linspace(V_range[0], V_range[1], num_points)
        I_data = self.composite_model(V_data, *params)
        I_data += np.random.normal(0, noise_level, size=I_data.shape)

        return V_data, I_data


def analyze_junction_iv(V_data, I_data, model="composite", plot=True):
    """
    Convenience function for quick I-V analysis.

    Parameters
    ----------
    V_data : array_like
        Voltage data (V)
    I_data : array_like
        Current data (A)
    model : str, default 'composite'
        Tunneling model to fit
    plot : bool, default True
        Whether to plot results

    Returns
    -------
    dict
        Analysis results including fitted parameters
    """
    analyzer = JosephsonJunctionAnalyzer()

    try:
        popt, pcov = analyzer.fit_iv_data(V_data, I_data, model=model)

        if plot:
            analyzer.plot_fit_results(V_data, I_data, popt, model=model)

        # Calculate parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))

        # Calculate R-squared
        if model == "composite":
            I_fit = analyzer.composite_model(V_data, *popt)
        elif model == "direct":
            I_fit = analyzer.direct_tunneling(V_data, *popt)
        elif model == "trap_assisted":
            I_fit = analyzer.trap_assisted_tunneling(V_data, *popt)
        elif model == "fowler_nordheim":
            I_fit = analyzer.fowler_nordheim(V_data, *popt)

        ss_res = np.sum((I_data - I_fit) ** 2)
        ss_tot = np.sum((I_data - np.mean(I_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            "fitted_parameters": popt,
            "parameter_errors": param_errors,
            "covariance_matrix": pcov,
            "r_squared": r_squared,
            "model": model,
            "analyzer": analyzer,
        }

    except Exception as e:
        print(f"Fitting failed: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = JosephsonJunctionAnalyzer()

    # Generate synthetic data for demonstration
    V_data, I_data = analyzer.generate_synthetic_data()

    # Analyze the data
    results = analyze_junction_iv(V_data, I_data, model="composite")

    if results:
        print("Fit successful!")
        print(f"R-squared: {results['r_squared']:.4f}")
        print(f"Fitted parameters: {results['fitted_parameters']}")
