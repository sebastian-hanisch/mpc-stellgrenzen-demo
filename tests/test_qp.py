"""Einheiten von msg_qp.py an Handbeispielen: kondensierte Matrizen, Aktivmengen-Methode, Spur, Rückfall (DEMO-PLAYBOOK §1b)."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import msg_dynamics as D
import msg_qp as QP
from conftest import hessian_by_differences


def _wagen_hand():
    A, B, Q = D.plant("wagen")
    return QP.condensed(2, 1.0, np.eye(2), A, B, Q), A, B, Q


def test_condensed_matrices_match_the_hand_values():
    """Von Hand (Wagen, N = 2, Q = P = I, R = 1): Γ = [[0,0],[1,0],[1,0],[1,1]], H = [[8,2],[2,4]], f = F x0 = (−2, −4) bei x0 = (7, −2), c = x0'(Q + Φ'Q̄Φ)x0."""
    (Phi, G, H, F, C), A, B, Q = _wagen_hand()
    assert np.array_equal(G, [[0, 0], [1, 0], [1, 0], [1, 1]])
    assert np.array_equal(H, [[8, 2], [2, 4]])
    assert np.array_equal(F @ np.array([7.0, -2.0]), [-2.0, -4.0])
    x0 = np.array([7.0, -2.0])
    # c: 53 (= x0'x0) + |x1|² + |x2|² mit u = 0: x1 = (5, −2), x2 = (3, −2): 53 + 29 + 13 = 95
    assert x0 @ C @ x0 == pytest.approx(95.0)


def test_condensed_matches_exact_differences_of_the_direct_cost():
    """H, f, c aus der direkt summierten Kostenfunktion (Brüche, Differenzen) gegen die Matrixformel, N = 1 bis 4 mit P = I und einem anderen P."""
    A, B, Q = D.plant("wagen")
    for N in (1, 2, 3, 4):
        for P in (np.eye(2), np.array([[3.0, 1.0], [1.0, 2.0]])):
            Pf = [[Fr(int(v)) for v in row] for row in P]
            x0 = (Fr(7), Fr(-2))
            Hd, fd, J0 = hessian_by_differences(N, x0, Fr(1), Pf)
            Phi, G, H, F, C = QP.condensed(N, 1.0, P, A, B, Q)
            xa = np.array([7.0, -2.0])
            assert np.allclose(H, np.array(Hd, float))
            assert np.allclose(F @ xa, np.array(fd, float))
            assert xa @ C @ xa == pytest.approx(float(J0))


def test_active_set_finds_the_hand_solution():
    """Handbeispiel: unbeschränkt (0, 1), mit Box ±½ die Lösung (1/8, 1/2); u₁ liegt oben mit g₁ = −7/4 < 0, u₀ ist frei."""
    (Phi, G, H, F, C), A, B, Q = _wagen_hand()
    f = F @ np.array([7.0, -2.0])
    r = QP.qp_active_set(H, f, -0.5 * np.ones(2), 0.5 * np.ones(2))
    assert r.u == pytest.approx([0.125, 0.5], abs=1e-12)
    assert not r.fallback
    g = H @ r.u + f
    assert g == pytest.approx([0.0, -1.75], abs=1e-12)


def test_unconstrained_case_takes_a_full_step_and_stops():
    """Ohne aktive Grenze: Durchlauf 1 ist der volle Newton-Schritt, Durchlauf 2 stellt die Optimalität fest (2 Iterationen)."""
    (Phi, G, H, F, C), *_ = _wagen_hand()
    f = F @ np.array([7.0, -2.0])
    r = QP.qp_active_set(H, f, -5 * np.ones(2), 5 * np.ones(2))
    assert r.u == pytest.approx([0.0, 1.0], abs=1e-12)
    assert r.iterations == 2


def test_zero_gradient_start_needs_one_iteration():
    """x0 = 0: f = 0, u = 0 ist schon optimal, ein Durchlauf."""
    (Phi, G, H, F, C), *_ = _wagen_hand()
    r = QP.qp_active_set(H, np.zeros(2), -np.ones(2), np.ones(2))
    assert r.iterations == 1 and np.all(r.u == 0)


def test_trace_records_working_set_and_value():
    (Phi, G, H, F, C), *_ = _wagen_hand()
    f = F @ np.array([7.0, -2.0])
    r = QP.qp_active_set(H, f, -0.5 * np.ones(2), 0.5 * np.ones(2), trace=True)
    assert len(r.trace) == r.iterations
    assert r.trace[0]["iteration"] == 1 and r.trace[0]["wert"] == pytest.approx(0.0)      # Start u = 0, Wert ½·0 + 0
    assert r.trace[-1]["aktion"] == "optimal"
    assert any(s["oben"][1] for s in r.trace)                                               # u₁ wird irgendwann an die obere Grenze gelegt
    werte = [s["wert"] for s in r.trace]
    assert all(b <= a + 1e-12 for a, b in zip(werte, werte[1:]))                           # der Zielwert fällt nie


def test_degenerate_box_returns_the_fixed_point():
    """lo = hi: nichts ist frei, die Lösung ist der Punkt selbst."""
    r = QP.qp_active_set(np.eye(3), np.array([1.0, 2.0, 3.0]), 0.5 * np.ones(3), 0.5 * np.ones(3))
    assert np.all(r.u == 0.5)


def test_box_not_containing_zero_starts_on_the_bound():
    """lo > 0: Start bei u = lo; min ½u² + 5u mit u ∈ [1, 2] liegt bei 1."""
    r = QP.qp_active_set(np.array([[1.0]]), np.array([5.0]), np.array([1.0]), np.array([2.0]))
    assert r.u[0] == pytest.approx(1.0)


def test_sign_pattern_marks_the_bounds():
    assert QP.sign_pattern([-1.0, 0.2, 1.0, 1.0 - 1e-12], -np.ones(4), np.ones(4)) == (-1, 0, 1, 1)


def test_fallback_is_used_when_the_active_set_method_gives_up(monkeypatch):
    """Bricht die Aktivmengen-Methode ab, gilt der Rückfall (bvls): gleiche Lösung, `fallback=True`, `iterations=-1`."""
    (Phi, G, H, F, C), *_ = _wagen_hand()
    f = F @ np.array([7.0, -2.0])

    def boom(*a, **k):
        raise QP.QPNoConvergence("test")

    monkeypatch.setattr(QP, "qp_active_set", boom)
    r = QP.qp_solve(H, f, -0.5 * np.ones(2), 0.5 * np.ones(2))
    assert r.fallback and r.iterations == -1
    assert r.u == pytest.approx([0.125, 0.5], abs=1e-9)


def test_maxit_raises_instead_of_looping():
    (Phi, G, H, F, C), *_ = _wagen_hand()
    f = F @ np.array([7.0, -2.0])
    with pytest.raises(QP.QPNoConvergence):
        QP.qp_active_set(H, f, -0.5 * np.ones(2), 0.5 * np.ones(2), maxit=1)


def test_qp_value_matches_the_direct_cost_of_the_sequence():
    """½u'Hu + f'u + c ist die direkt gerechnete Kostensumme der Folge (Handbeispiel: MPC 1495/16)."""
    (Phi, G, H, F, C), *_ = _wagen_hand()
    x0 = np.array([7.0, -2.0])
    assert QP.qp_value(np.array([0.125, 0.5]), H, F @ x0, float(x0 @ C @ x0)) == pytest.approx(1495 / 16)
