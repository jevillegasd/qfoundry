"""
qfoundry: utilities for superconducting circuit design and analysis.

Public API (lazy-imported):
- resonator: cpw, cpw_resonator
- qubits: transmon, tunable_transmon
- simulation: capacitance

Versioning follows PEP 440; see __version__.
"""

from importlib.metadata import version, PackageNotFoundError
from .resonator import cpw, cpw_resonator
from .qubits import transmon, tunable_transmon

try:
    __version__ = version("qfoundry")
except PackageNotFoundError:
    # Fallback for editable or source usage without installed metadata
    __version__ = "0.1.0"

__all__ = [
    "cpw",
    "cpw_resonator",
    "transmon",
    "tunable_transmon",
    "capacitance",
    "josephson",
    "__version__",
]


def __getattr__(name):
    if name in {"cpw", "cpw_resonator"}:
        from .resonator import cpw, cpw_resonator  # type: ignore

        return {"cpw": cpw, "cpw_resonator": cpw_resonator}[name]
    if name in {"transmon", "tunable_transmon"}:
        from .qubits import transmon, tunable_transmon  # type: ignore

        return {"transmon": transmon, "tunable_transmon": tunable_transmon}[name]
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
    raise AttributeError(name)
