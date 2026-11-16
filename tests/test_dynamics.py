"""Einheiten von msg_dynamics.py an Handbeispielen: Strecken, Riccati-Gewinn, Regeln, Regelkreis, Zählung, Gitter."""

import numpy as np
import pytest

import msg_constants as C
import msg_dynamics as D


def test_plants():
    A, B, Q = D.plant("wagen", 1.2)
    assert np.array_equal(A, [[1.2, 1.0], [0.0, 1.0]]) and np.array_equal(B, [[0.0], [1.0]]) and np.array_equal(Q, np.eye(2))
    A, B, Q = D.plant("skalar")
    assert A.shape == (1, 1) and A[0, 0] == B[0, 0] == Q[0, 0] == 1.0
    with pytest.raises(ValueError):
        D.plant("kugel")


def test_dare_gain_of_the_wagen_for_r_equal_one():
    """V1 der Messung: K = (0,42208; 1,24393) für R = 1; geschlossene Eigenwerte 0,378 ± 0,188 i (Betrag 0,4224)."""
    A, B, Q = D.plant("wagen")
    P, K = D.dare(A, B, Q, 1.0)
    assert K[0] == pytest.approx([0.42208, 1.24393], abs=5e-6)
    assert np.abs(np.linalg.eigvals(A - B @ K)) == pytest.approx([0.4224, 0.4224], abs=5e-4)
    assert (-K @ np.array([10.0, 0.0]))[0] == pytest.approx(-4.2208, abs=1e-4)             # „verlangt u = −4,22“


def test_clip_law_clips_symmetrically():
    law = D.ClipLaw(np.array([[0.5, 1.0]]), 1.0)
    assert law(np.array([10.0, 0.0])) == -1.0 and law(np.array([-10.0, 0.0])) == 1.0 and law(np.array([1.0, 0.5])) == pytest.approx(-1.0)
    assert law(np.array([0.2, 0.0])) == pytest.approx(-0.1)


def test_run_loop_cost_by_hand_in_the_scalar():
    """Skalar x⁺ = x + u mit u = −x/2 (K = 1/2), x0 = 4, Kosten Σ(x² + u²) = Σ 1,25 x² mit x_k = 4/2^k: 1,25 · 16 / (1 − 1/4) = 80/3 (ohne Faktor ½)."""
    A, B, Q = D.plant("skalar")
    lp = D.run_loop(D.ClipLaw(np.array([[0.5]]), 100.0), np.array([4.0]), A, B, Q, 1.0, T=200, tol=1e-14)
    assert lp.cost == pytest.approx(80 / 3, rel=1e-12)
    assert lp.steps > 30 and lp.end_norm < 1e-14
    lp2 = D.run_loop(D.ClipLaw(np.array([[0.5]]), 100.0), np.array([4.0]), A, B, Q, 1.0, T=3, record=True)
    assert lp2.xs.shape == (4, 1) and lp2.us == pytest.approx([-2.0, -1.0, -0.5]) and lp2.xs[:, 0] == pytest.approx([4.0, 2.0, 1.0, 0.5])
    assert lp2.cost == pytest.approx(1.25 * (16 + 4 + 1))


def test_run_loop_stops_on_blow_up():
    A, B, Q = D.plant("skalar")
    lp = D.run_loop(lambda x: x[0], np.array([1.0]), A, B, Q, 1.0, T=400)     # x⁺ = 2x wächst
    assert lp.end_norm > 1e12 and lp.steps < 60


def test_compare_counts_by_hand():
    """Kosten 10, 10, 10 (clip) gegen 9, 10, 11 (MPC): billiger 1, gleich 1, teurer 1, größter Vorteil 10 %, größter Nachteil 10 %. Gleich heißt relativ höchstens 1e-9."""
    ch, eq, de_, adv, dis = D.compare_counts([10.0, 10.0, 10.0], [9.0, 10.0 * (1 + 1e-12), 11.0])
    assert (ch, eq, de_) == (1, 1, 1) and adv == pytest.approx(10.0) and dis == pytest.approx(10.0)
    assert D.compare_counts([], []) == (0, 0, 0, 0.0, 0.0)


def test_grid_points_drop_the_origin():
    pts = D.grid_points(C.GRID_X1, C.GRID_X2)
    assert len(pts) == 21 * 17 - 1 == 356 and (0.0, 0.0) not in pts
    assert len(D.grid_points(C.UNSTABLE_GRID, C.UNSTABLE_GRID)) == 624
    assert len(D.grid_points(C.FIRSTU_X1, C.FIRSTU_X2, drop_origin=False)) == 41 * 33 == 1353
    assert len(D.grid_axis(C.SCALAR_GRID)) == 2401


def test_mpc_with_n1_equals_clip_in_cost():
    """N = 1: dasselbe wie Abschneiden (Kosten gleich auf allen Gitterpunkten, hier eine Stichprobe)."""
    prob = D.build_mpc("wagen", 1.0, 1, 1.0, 1.0, "DARE")
    pts = D.grid_points((-10.0, 10.0, 4.0), (-4.0, 4.0, 2.0))
    g = D.run_grid(prob, pts)
    assert np.allclose(g.cost_clip, g.cost_mpc, rtol=1e-9)


def test_mpc_law_counts_calls_iterations_and_patterns():
    prob = D.build_mpc("wagen", 1.0, 5, 1.0, 1.0, "DARE")
    law = D.MPCLaw(prob, record=True)
    D.run_loop(law, np.array([10.0, 0.0]), prob.A, prob.B, prob.Q, 1.0)
    assert law.calls == len(law.iterations) + law.fallbacks and law.calls > 10 and law.fallbacks == 0
    assert min(law.iterations) >= 1 and len(law.patterns) >= 3
    assert (-1,) * 5 in law.patterns or any(p[0] == -1 for p in law.patterns)    # der erste Takt ist bei Start (10; 0) an der unteren Grenze


def test_unconstrained_first_gain_with_dare_endcost_equals_minus_k():
    """Mit P = DARE ist die erste Stelle der unbeschränkten N-Schritt-Lösung für jedes N gleich −K x."""
    for N in (1, 2, 5, 10):
        prob = D.build_mpc("wagen", 1.0, N, 1.0, 1.0, "DARE")
        assert D.unconstrained_first_gain(prob) == pytest.approx(-prob.K[0], abs=1e-12)


def test_converges_on_known_cases():
    A, B, Q = D.plant("wagen")
    P, K = D.dare(A, B, Q, 1.0)
    assert D.converges(D.ClipLaw(K, 1.0), (3.0, 1.0), A, B)
    assert not D.converges(lambda x: 0.0, (3.0, 1.0), A, B)          # ohne Eingriff bleibt der Doppelintegrator in Bewegung (x₂ bleibt 1)
    A2, B2, _ = D.plant("wagen", 1.2)
    assert not D.converges(lambda x: 0.0, (3.0, 1.0), A2, B2)        # a = 1,2 ohne Eingriff wächst über 1e6


def test_controllable_mask_by_hand():
    """Doppelintegrator mit |u| ≤ 1 in 100 Schritten: der Ursprung und kleine Zustände sind steuerbar; x₁ = 10⁶ nicht (Beschleunigen und Bremsen braucht etwa 2·√(10⁶) = 2 000 Schritte)."""
    A, B, _ = D.plant("wagen")
    m = D.controllable_mask(A, B, [(0.0, 0.0), (5.0, 0.0), (1e6, 0.0)])
    assert list(m) == [True, True, False]
