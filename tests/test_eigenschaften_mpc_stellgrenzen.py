"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen über das Verfahren, die ohne bekannte Lösung gelten müssen. Je Eigenschaft 20 bis 50 Instanzen, Zufalls-Seed fest (SplitMix64).

Symmetrie der Box, Skalierung von Zustand und Grenze, Kosten nie unter x'P x, größere Grenze senkt den QP-Wert, u_max → ∞ liefert −Kx,
N = 1: QP = abgeschnitten (für beliebiges P), Zielfaktor c > 0 ändert die Lösung nicht, Umnummerieren ist nicht möglich (die Zeitachse ist geordnet)."""

import numpy as np
import pytest

import msg_dynamics as D
import msg_qp as QP
from msg_rng import SplitMix64


def _state(rng, s1=10.0, s2=4.0):
    return np.array([-s1 + 2 * s1 * rng.random(), -s2 + 2 * s2 * rng.random()])


def test_symmetry_of_the_law():
    """Die Box ist symmetrisch, also κ(−x) = −κ(x) (30 Zustände, N = 6, u_max = 0,75)."""
    rng = SplitMix64(30)
    law = D.MPCLaw(D.build_mpc("wagen", 1.0, 6, 1.0, 0.75, "DARE"))
    for _ in range(30):
        x = _state(rng)
        assert law(-x) == pytest.approx(-law(x), abs=1e-9)


def test_scaling_of_state_and_limit():
    """Kosten(c·x0, c·u_max) = c²·Kosten(x0, u_max) für Abschneiden und MPC (20 Instanzen)."""
    rng = SplitMix64(31)
    for _ in range(20):
        x0 = _state(rng, 8.0, 3.0)
        c = 0.3 + 4 * rng.random()
        for mk in ("clip", "mpc"):
            costs = []
            for um, sc in ((1.0, 1.0), (c, c)):
                prob = D.build_mpc("wagen", 1.0, 6, 1.0, um, "DARE")
                law = D.ClipLaw(prob.K, um) if mk == "clip" else D.MPCLaw(prob)
                costs.append(D.run_loop(law, sc * x0, prob.A, prob.B, prob.Q, 1.0).cost)
            assert costs[1] == pytest.approx(c * c * costs[0], rel=1e-9)


def test_cost_never_below_the_unconstrained_value():
    """Jede zulässige Regel kostet mindestens x0'P x0 (Wert des unbeschränkten LQR); 30 Zustände, beide Regeln."""
    rng = SplitMix64(32)
    prob = D.build_mpc("wagen", 1.0, 5, 1.0, 0.5, "DARE")
    for _ in range(30):
        x0 = _state(rng)
        low = float(x0 @ prob.P @ x0)
        assert D.run_loop(D.MPCLaw(prob), x0, prob.A, prob.B, prob.Q, 1.0).cost >= low * (1 - 1e-9)
        assert D.run_loop(D.ClipLaw(prob.K, 0.5), x0, prob.A, prob.B, prob.Q, 1.0).cost >= low * (1 - 1e-9)


def test_larger_limit_never_raises_the_qp_value():
    rng = SplitMix64(33)
    for _ in range(40):
        N = 2 + int(rng.below(8))
        x0 = _state(rng)
        vals = []
        for um in (0.25, 0.5, 1.0, 2.0, 50.0):
            p = D.build_mpc("wagen", 1.0, N, 1.0, um, "DARE")
            u = QP.qp_solve(p.H, p.F @ x0, p.lo, p.hi).u
            vals.append(QP.qp_value(u, p.H, p.F @ x0, float(x0 @ p.Cmat @ x0)))
        assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))


def test_huge_limit_gives_the_unconstrained_gain():
    """u_max → ∞: das erste Stellsignal ist −K x (Endkosten DARE, jeder Horizont)."""
    rng = SplitMix64(34)
    for _ in range(30):
        N = 1 + int(rng.below(10))
        x0 = _state(rng)
        p = D.build_mpc("wagen", 1.0, N, 1.0, 1e6, "DARE")
        assert QP.qp_solve(p.H, p.F @ x0, p.lo, p.hi).u[0] == pytest.approx(-(p.K @ x0)[0], abs=1e-6)


def test_n1_equals_clip_for_any_terminal_matrix():
    """N = 1: das Box-QP ist die abgeschnittene unbeschränkte Lösung, auch mit anderem P (zufällige positiv definite P, 30 Instanzen)."""
    rng = SplitMix64(35)
    A, B, Q = D.plant("wagen")
    for _ in range(30):
        M = np.array([[rng.normal(), rng.normal()], [rng.normal(), rng.normal()]])
        P = M @ M.T + 0.1 * np.eye(2)
        _, _, H, F, _ = QP.condensed(1, 0.5 + rng.random(), P, A, B, Q)
        x0 = _state(rng)
        um = 0.2 + 1.8 * rng.random()
        u = QP.qp_active_set(H, F @ x0, np.array([-um]), np.array([um])).u[0]
        assert u == pytest.approx(float(np.clip(-(F @ x0)[0] / H[0, 0], -um, um)), abs=1e-12)


def test_positive_factor_on_the_objective_does_not_change_the_solution():
    rng = SplitMix64(36)
    for _ in range(30):
        N = 2 + int(rng.below(6))
        p = D.build_mpc("wagen", 1.0, N, 1.0, 0.5, "DARE")
        f = p.F @ _state(rng)
        c = 0.01 + 100 * rng.random()
        assert QP.qp_active_set(c * p.H, c * f, p.lo, p.hi).u == pytest.approx(QP.qp_active_set(p.H, f, p.lo, p.hi).u, abs=1e-8)


def test_same_inputs_give_the_same_output():
    """Kein Zufall in der Rechnung: zweimal derselbe Aufruf, bitgleich."""
    p = D.build_mpc("wagen", 1.0, 8, 1.0, 1.0, "DARE")
    x0 = np.array([7.3, -1.1])
    a = D.run_loop(D.MPCLaw(p), x0, p.A, p.B, p.Q, 1.0)
    b = D.run_loop(D.MPCLaw(p), x0, p.A, p.B, p.Q, 1.0)
    assert a.cost == b.cost and a.steps == b.steps


def test_mpc_never_dearer_than_clip_on_random_states_with_dare_endcost():
    """Gegenprobe (Aufzählung „nie teurer“, kleine Fassung der 1 000 Instanzen): 40 Zufallsinstanzen mit anderer Verteilung als das Gitter."""
    rng = SplitMix64(14)
    for _ in range(40):
        R = (0.1, 1.0, 10.0)[rng.below(3)]
        um = 0.2 + 1.8 * rng.random()
        x0 = np.array([-30.0 + 60.0 * rng.random(), -12.0 + 24.0 * rng.random()])
        N = (2, 3, 5)[rng.below(3)]
        p = D.build_mpc("wagen", 1.0, N, R, um, "DARE")
        c_clip = D.run_loop(D.ClipLaw(p.K, um), x0, p.A, p.B, p.Q, R, T=3000)
        c_mpc = D.run_loop(D.MPCLaw(p), x0, p.A, p.B, p.Q, R, T=3000)
        assert c_mpc.end_norm < 1e-6 and c_mpc.cost <= c_clip.cost * (1 + 1e-9)
