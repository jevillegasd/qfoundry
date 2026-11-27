"""Regression tests for the Schur-reduced, mode-projected capacitances of a
floating two-island transmon (qfoundry.utils.effective_coupling /
effective_Cq and bus_resonator_coupler.from_maxwell).

Reference data: FEM Maxwell matrix "333" (family type_A, fF) of design
qf04h007 — a floating transmon with readout, feedline, XY drive and two
coupler ports. The readout coupling derived from this matrix was validated
against measurement on qpu191: the raw island-2-to-readout entry (6.79 fF)
overestimates g by ~2.3x; the differential Schur value (3.07 fF) gives the
measured order of magnitude.
"""
import numpy as np
import pytest

from qfoundry.utils import (
    Cs_to_E, effective_coupling, effective_Cq, mode_weights, reduce_maxwell,
)

NODES_333 = ["Island 1", "Island 2", "Readout", "Feedline", "Ctrl XY", "Coupling 1", "Coupling 2"]
# Maxwell convention, fF (diagonal = total node self-capacitance, off-diagonals negated mutuals)
M_333_fF = np.array([
    [124.4,     -34.613,   -0.64031,  -0.018266, -0.06515,  -8.5275,   -8.4183],
    [-34.613,   126.22,    -6.7872,   -0.034262, -0.13224,  -0.50388,  -0.44144],
    [-0.64019,  -6.7874,   683.52,    -6.6684,   -0.15087,  -0.040387, -0.038199],
    [-0.01827,  -0.034215, -6.6686,   377.92,    -0.019666, -0.0019352, -0.0018954],
    [-0.065186, -0.13228,  -0.15095,  -0.019675, 140.92,    -0.003461, -0.010628],
    [-8.5276,   -0.50388,  -0.040414, -0.0019384, -0.0034692, 185.25,  -0.038855],
    [-8.4183,   -0.44142,  -0.038007, -0.0018585, -0.010634, -0.038889, 185.38],
])
ISLANDS = [0, 1]
READOUT, COUPLER1, COUPLER2 = 2, 5, 6


def test_floating_readout_coupling_is_differential_schur_value():
    """Cg_eff = |C_2R - C_1R|/2 of the reduced islands+readout block = 3.07 fF,
    not the raw island-2 entry 6.79 fF nor C_1R + C_2R = 7.4 fF."""
    Cg = effective_coupling(M_333_fF, ISLANDS, READOUT)
    assert Cg == pytest.approx(3.07, abs=0.01)
    S = reduce_maxwell(M_333_fF, ISLANDS + [READOUT])
    assert Cg == pytest.approx(abs(S[1, 2] - S[0, 2]) / 2.0, rel=1e-9)
    assert Cg < 0.5 * abs(M_333_fF[1, READOUT])


def test_floating_coupler_port_coupling():
    """Qubit-side bus capacitance Cc = 4.01 fF — the raw Island1–Coupling1
    entry (8.53 fF) is ~2.1x too large (→ ~4.5x in J)."""
    Cc1 = effective_coupling(M_333_fF, ISLANDS, COUPLER1)
    Cc2 = effective_coupling(M_333_fF, ISLANDS, COUPLER2)
    assert Cc1 == pytest.approx(4.01, abs=0.01)
    assert Cc2 == pytest.approx(3.99, abs=0.01)
    assert abs(M_333_fF[0, COUPLER1]) / Cc1 == pytest.approx(2.125, abs=0.03)


def test_floating_Cq_and_Ec_match_schur_reduction():
    """C_Sigma = C_12 + C_1g C_2g/(C_1g + C_2g) of the reduced island block:
    79.77 fF → Ec/h = 242.8 MHz (the stale single-island pipeline gave
    73.35 fF / 264.1 MHz)."""
    Cq = effective_Cq(M_333_fF, ISLANDS)
    assert Cq == pytest.approx(79.77, abs=0.02)
    assert Cs_to_E(Cq * 1e-15) / 1e6 == pytest.approx(242.8, abs=0.2)


def test_grounded_definitions_reduce_to_plain_schur():
    """A single node: coupling = reduced mutual, C_Sigma = reduced diagonal."""
    S = reduce_maxwell(M_333_fF, [0, READOUT])
    assert effective_coupling(M_333_fF, 0, READOUT) == pytest.approx(abs(S[0, 1]))
    assert effective_Cq(M_333_fF, 0) == pytest.approx(reduce_maxwell(M_333_fF, [0])[0, 0])
    assert list(mode_weights(0)) == [1.0]
    assert list(mode_weights([0, 1])) == [0.5, -0.5]


def test_symmetric_islands_give_exact_half_difference():
    """For perfectly symmetric islands the differential projection is exactly
    (C_2t - C_1t)/2 and the common-mode part is screened out."""
    M = np.array([
        [100.0, -30.0, -2.0, -0.5],
        [-30.0, 100.0, -6.0, -0.5],
        [-2.0,  -6.0,  50.0, -1.0],
        [-0.5,  -0.5,  -1.0, 40.0],
    ])
    # Node 3 kept out of the reduction target: it is eliminated.
    S = reduce_maxwell(M, [0, 1, 2])
    assert effective_coupling(M, [0, 1], 2) == pytest.approx((abs(S[1, 2]) - abs(S[0, 2])) / 2)
    # Equal coupling of both islands to the electrode → pure common mode → zero.
    M2 = M.copy()
    M2[0, 2] = M2[2, 0] = -4.0
    M2[1, 2] = M2[2, 1] = -4.0
    assert effective_coupling(M2, [0, 1], 2) == pytest.approx(0.0, abs=1e-12)


def test_keep_leaves_other_dynamical_nodes_in_the_network():
    """With ``keep`` the other qubit is not folded into the coupling (3-node
    grounded bus edge: couplings are the plain matrix entries)."""
    M = np.array([[90.0, -8.0, -0.15], [-8.0, 200.0, -8.5], [-0.15, -8.5, 85.0]])
    assert effective_coupling(M, 0, 1, keep=[2]) == pytest.approx(8.0)
    assert effective_coupling(M, 2, 1, keep=[0]) == pytest.approx(8.5)
    assert effective_coupling(M, 0, 2, keep=[1]) == pytest.approx(0.15)
    # Without keep the path through node 2 is folded in (different number).
    assert effective_coupling(M, 0, 1) != pytest.approx(8.0, rel=1e-6)


class _Q:
    """Minimal duck-typed qubit for the edge constructors."""

    def __init__(self, f01, C):
        self._f, self._C = f01, C

    def f01(self):
        return self._f

    def C(self):
        return self._C

    def omega01(self):
        return 2 * np.pi * self._f

    def alpha(self):
        return -200e6


def test_from_maxwell_grounded_regression():
    """Single-node qubits: unchanged C_0r/C_1r/C_01 and mode capacitance."""
    from qfoundry.edges import bus_resonator_coupler

    nodes = ["Q0", "RES", "Q1"]
    M_F = np.array([[90.0, -8.0, -0.15], [-8.0, 200.0, -8.5], [-0.15, -8.5, 85.0]]) * 1e-15
    cp = bus_resonator_coupler.from_maxwell(_Q(5e9, 80e-15), _Q(5.2e9, 80e-15),
                                            nodes, M_F, "Q0", "RES", "Q1", 200e9)
    assert cp.C_0r * 1e15 == pytest.approx(8.0, rel=1e-9)
    assert cp.C_1r * 1e15 == pytest.approx(8.5, rel=1e-9)
    assert cp.C_01 * 1e15 == pytest.approx(0.15, rel=1e-9)
    C_mode_fF = (200.0 - 8.0 - 8.5) / 2.0 + 8.0 + 8.5
    assert Cs_to_E(cp.resonator.C()) == pytest.approx(Cs_to_E(C_mode_fF * 1e-15), rel=1e-3)


def test_from_maxwell_floating_qubits_use_differential_coupling():
    """Two floating transmons (island pairs from matrix 333) sharing a bus:
    C_0r/C_1r are the differential |C_2r - C_1r|/2 (~4.0 fF), while naming
    a single island returns the raw-like ~8.5 fF."""
    from qfoundry.edges import bus_resonator_coupler

    nodes = ["q0i1", "q0i2", "BUS", "q1i1", "q1i2"]
    M = np.zeros((5, 5))
    for a, b in ((0, 1), (3, 4)):
        M[a, a], M[b, b] = 124.4, 126.2
        M[a, b] = M[b, a] = -34.61
        M[a, 2] = M[2, a] = -8.53
        M[b, 2] = M[2, b] = -0.50
    M[2, 2] = 185.0
    M[0, 3] = M[3, 0] = -0.2  # small direct island-to-island (geometric) capacitance
    M_F = M * 1e-15
    q0, q1 = _Q(5e9, 80e-15), _Q(5.2e9, 80e-15)
    fl = bus_resonator_coupler.from_maxwell(q0, q1, nodes, M_F, ("q0i1", "q0i2"), "BUS",
                                            ("q1i1", "q1i2"), 200e9)
    assert fl.C_0r * 1e15 == pytest.approx((8.53 - 0.50) / 2, rel=1e-9)
    assert fl.C_1r * 1e15 == pytest.approx((8.53 - 0.50) / 2, rel=1e-9)
    # Direct C_01 is the *geometric* island-island term projected on both
    # differential modes (the bus stays dynamical, so the through-resonator
    # path is not folded in): (C_11 - C_12 - C_21 + C_22)/4 = 0.2/4.
    assert fl.C_01 * 1e15 == pytest.approx(0.05, rel=1e-9)
    gr = bus_resonator_coupler.from_maxwell(q0, q1, nodes, M_F, "q0i1", "BUS", "q1i1", 200e9)
    assert gr.C_0r / fl.C_0r == pytest.approx(2.16, abs=0.05)
    # J scales with C_0r·C_1r (up to the slightly different bus mode
    # capacitance): the single-island reading is ~4.7x too large.
    assert abs(gr.g_bus()) / abs(fl.g_bus()) == pytest.approx(
        (gr.C_0r * gr.C_1r) / (fl.C_0r * fl.C_1r), rel=0.02)
    assert abs(gr.g_bus()) / abs(fl.g_bus()) > 4.0
    with pytest.raises(ValueError):
        bus_resonator_coupler.from_maxwell(q0, q1, nodes, M_F, ("q0i1", "nope"), "BUS", "q1i1", 200e9)
