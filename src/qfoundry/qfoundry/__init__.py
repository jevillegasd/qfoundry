"""
qfoundry: utilities for superconducting circuit design and analysis.

Public API (lazy-imported):
- resonator: cpw, cpw_resonator
- qubits: transmon, tunable_transmon
- simulation: capacitance

Versioning follows PEP 440; see __version__.
"""

from importlib.metadata import version, PackageNotFoundError
from .circuit import circuit
from .resonator import cpw, cpw_resonator
from .qubits import transmon, tunable_transmon
from .waveguides import cpw

try:
    __version__ = version("qfoundry")
except PackageNotFoundError:
    # Fallback for editable or source usage without installed metadata
    __version__ = "0.1.0"

__all__ = [
    "circuit",
    "cpw",
    "cpw_resonator",
    "transmon",
    "tunable_transmon",
    "qubit",
    "capacitance",
    "josephson",
    "edge",
    "capacitive_coupler",
    "inductive_coupler",
    "bus_resonator_coupler",
    "tunable_coupler",
    "hybrid_coupler",
    "__version__",
]


def __getattr__(name):
    if name in {"cpw", "cpw_resonator"}:
        from .resonator import cpw_resonator  # type: ignore
        from .waveguides import cpw  # type: ignore
        return {"cpw": cpw, "cpw_resonator": cpw_resonator}[name]
    if name in {"transmon", "tunable_transmon", "qubit"}:
        from .qubits import transmon, tunable_transmon, qubit  # type: ignore

        return {"transmon": transmon, "tunable_transmon": tunable_transmon, "qubit": qubit}[name]
    if name == "capacitance":
        try:
            from .simulation import capacitance  # type: ignore
        except Exception as exc:
            raise ImportError(
                "qfoundry simulation extras are not installed. Install with: pip install 'qfoundry[simulation]'"
            ) from exc
        return capacitance
    if name == "josephson":
        from . import josephson  # type: ignore

        return josephson
    if name in {
        "edge",
        "capacitive_coupler",
        "inductive_coupler",
        "bus_resonator_coupler",
        "tunable_coupler",
        "hybrid_coupler",
    }:
        from .edges import (  # type: ignore
            edge,
            capacitive_coupler,
            inductive_coupler,
            bus_resonator_coupler,
            tunable_coupler,
            hybrid_coupler,
        )
        return locals()[name]
    raise AttributeError(name)
