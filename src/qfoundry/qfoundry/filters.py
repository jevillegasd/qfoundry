""" Filter synthesis using CPWs """

from cmath import sqrt
from qfoundry.waveguides import cpw
from qfoundry.PDK import QF_PDK
from typing import Literal

def circuit_element(C=0, L=0):
    """ Simple representation of a circuit element """
    return {'C': C, 'L': L}


def lowpass_prototype(order):
    """ Return a lowpass prototype filter element values as a list """
    g = []
    if order < 1:
        raise ValueError("Order must be at least 1")
    for n in range(1, order + 1):
        if n == 1:
            g.append(1.0)
        else:
            g_n = (4 * n - 2) / (n * (n - 1))
            g.append(g_n)
    return g

def bandpass_filter(start, end, order, ftype: Literal['tee', 'pi'] = 'tee', Z0=50):
    """ Create a lumped elemend butterworth bandpass filter (list of Capacitors and Inductors)
    Args:
        start (float): Start frequency in Hz
        end (float): End frequency in Hz
        order (int): Order of the filter
        ftype (str): Type of filter ('tee' or 'pi')
        impedance (float): Characteristic impedance of the filter in ohms
    Returns:
        dict: Dictionary of filter elements with their types and values

    The butterworth bandpass filter is created using a lowpass prototype transformation. For an order N filter, there are N resonators, each consisting of a series or shunt LC circuit depending on the filter type.
    Each capacitor and inductor value is calculated based on the bandwidth and center frequency of the desired bandpass filter using
    the following formulas:

    Reference:
    - Matthaei, Young, and Jones, "Microwave Filters, Impedance-Matching Networks, and Coupling Structures", 1980.
    
    """
    
    if ftype not in ['tee', 'pi']:
        raise NotImplementedError("Only 'tee' and 'pi' filter types are implemented")
    
    import math
    
    # Get lowpass prototype element values
    g = lowpass_prototype(order)
    
    # Calculate center frequency and bandwidth
    f0 = math.sqrt(start * end)  # Geometric mean
    BW = end - start
    FBW = BW / f0  # Fractional bandwidth
    omega0 = 2 * math.pi * f0
    
    # Create filter elements dictionary
    elements = {}
    
    for n in range(1, order + 1):
        g_n = g[n - 1]
        
        # Determine if this element is series or shunt based on filter type
        if ftype == 'tee':
            # Tee: odd elements are series, even elements are shunt
            is_series = (n % 2 == 1)
        else:  # pi
            # Pi: odd elements are shunt, even elements are series
            is_series = (n % 2 == 0)
        
        if is_series:
            # Series resonator: L in series with C
            L_series = (g_n * Z0) / (omega0 * FBW)
            C_series = FBW / (omega0 * g_n * Z0)
            elements[f'resonator_{n}'] = {
                'type': 'series',
                'element': circuit_element(C=C_series, L=L_series)
            }
        else:
            # Shunt resonator: L in parallel with C
            C_shunt = (g_n * FBW) / (omega0 * Z0)
            L_shunt = Z0 / (omega0 * FBW * g_n)
            elements[f'resonator_{n}'] = {
                'type': 'shunt',
                'element': circuit_element(C=C_shunt, L=L_shunt)
            }
    
    return elements




if __name__ == "__main__":
    # Example usage - Traditional lumped element filter
    print("Traditional Lumped Element Bandpass Filter (tee-type)")
    filter_elements = bandpass_filter(7.1e9, 7.8e9, 4, ftype='tee', Z0=50)
    for key, elem in filter_elements.items():
        print(f"{key}: Type={elem['type']}, C={elem['element']['C']*1e15:.2f} fF, L={elem['element']['L']*1e9:.2f} nH")
    
    waveguide: cpw = QF_PDK.cpw()
    print("\nWaveguide parameters:")
    print(f"  Width: {waveguide.w*1e6:.2f} um")
    print(f"  Gap: {waveguide.s*1e6:.2f} um")
    print(f"  Impedance: {waveguide.Z_0:.2f} Ohms")
    print(f"  Inductance per unit length: {waveguide.L*1e9:.2f} pH/mm")
    print(f"  Capacitance per unit length: {waveguide.C_m*1e12:.2f} fF/mm")

    