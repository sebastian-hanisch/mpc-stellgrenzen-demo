"""Orakel-Tests (anderer Rechenweg als die Demo, ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12):

* QP-Löser (Aktivmengen-Methode) gegen die Aufzählung aller 3^N Muster, gegen `lsq_linear(method="bvls", max_iter=100000)` (Cholesky-Umformung; die Voreinstellung der
  Iterationszahl ist bei N ≥ 20 zu ungenau: Zielwert bis 2,18 / 16,6 / 66,7 über dem Optimum) und gegen SLSQP (Toleranz 1e-4, nicht für N ≥ 60), gegen ein Stellfolgen-Gitter
  und gegen die Brüche-Aufzählung (N = 2);
* Riccati-Gewinn gegen die Riccati-Rekursion, kondensierte Kosten gegen die direkt summierte Kostenfunktion;
* Skalar: gefahrene Kosten gegen exakte Brüche und gegen dynamische Programmierung auf einem Gitter;
* steuerbare Menge (lineares Programm) gegen beschränkte kleinste Quadrate.
Zufall nur über den SplitMix64 der Demo (Zufalls-Seed 14), nie `numpy.random` oder `math.log`."""

import itertools
from fractions import Fraction as Fr

import numpy as np
import pytest

import msg_dynamics as D
import msg_exact as E
import msg_qp as QP
from conftest import cost_direct, enum_box_qp, hessian_by_differences
from msg_rng import SplitMix64


def _random_qp(rng, N):
    M = np.array([[rng.normal() for _ in range(N)] for _ in range(N)])
    H = M @ M.T + 0.1 * np.eye(N)
    f = np.array([3.0 * rng.normal() for _ in range(N)])
    lo = -np.array([0.2 + 1.8 * rng.random() for _ in range(N)])
    hi = np.array([0.2 + 1.8 * rng.random() for _ in range(N)])
    return H, f, lo, hi


def _value(H, f, u):
    return 0.5 * u @ H @ u + f @ u


def test_active_set_equals_full_enumeration_of_all_patterns():
    rng = SplitMix64(14)
    gleich = 0
    for i in range(120):
        N = 1 + i % 6
        H, f, lo, hi = _random_qp(rng, N)
        u = QP.qp_active_set(H, f, lo, hi).u
        ref, refv = enum_box_qp(H, f, lo, hi)
        assert np.max(np.abs(u - ref)) < 1e-9, (i, N)
        assert _value(H, f, u) == pytest.approx(refv, abs=1e-9)
        gleich += 1
    assert gleich == 120


def test_active_set_equals_bvls_and_slsqp():
    from scipy.optimize import minimize

    rng = SplitMix64(15)
    for i in range(40):
        N = 1 + i % 7
        H, f, lo, hi = _random_qp(rng, N)
        u = QP.qp_active_set(H, f, lo, hi).u
        assert np.max(np.abs(u - QP.qp_bvls(H, f, lo, hi))) < 1e-8
        s = minimize(lambda v: 0.5 * v @ H @ v + f @ v, np.clip(np.zeros(N), lo, hi), jac=lambda v: H @ v + f, bounds=list(zip(lo, hi)), method="SLSQP",
                     options={"ftol": 1e-15, "maxiter": 1000}).x
        assert _value(H, f, s) >= _value(H, f, u) - 1e-8          # SLSQP findet nie einen kleineren Wert
        assert np.max(np.abs(u - s)) < 1e-4


def test_mpc_qps_along_the_wagen_against_enumeration():
    rng = SplitMix64(16)
    for N in (2, 3, 4, 5, 6):
        prob = D.build_mpc("wagen", 1.0, N, 1.0, 1.0, "DARE")
        for _ in range(12):
            x0 = np.array([-10 + 20 * rng.random(), -4 + 8 * rng.random()])
            f = prob.F @ x0
            u = QP.qp_active_set(prob.H, f, prob.lo, prob.hi).u
            assert np.max(np.abs(u - enum_box_qp(prob.H, f, prob.lo, prob.hi)[0])) < 1e-9


def test_large_horizon_against_bvls_with_enough_iterations_and_kkt():
    """N = 10 und 20 (Konditionszahl 1,2·10³ und 1,4·10⁴): bvls mit max_iter = 100000 liegt höchstens 1e-8 im Wert über der eigenen Lösung; KKT-Verletzung klein."""
    rng = SplitMix64(17)
    for N in (10, 20):
        prob = D.build_mpc("wagen", 1.0, N, 1.0, 1.0, "DARE")
        for _ in range(6):
            x0 = np.array([-10 + 20 * rng.random(), -4 + 8 * rng.random()])
            f = prob.F @ x0
            u = QP.qp_solve(prob.H, f, prob.lo, prob.hi).u
            ub = QP.qp_bvls(prob.H, f, prob.lo, prob.hi)
            assert _value(prob.H, f, ub) - _value(prob.H, f, u) < 1e-8
            g = prob.H @ u + f
            innen = np.abs(u) < 1 - 1e-9
            assert np.max(np.abs(g[innen]), initial=0.0) < 1e-8
            assert np.max(np.maximum(0.0, -g[u <= -1 + 1e-9]), initial=0.0) < 1e-8
            assert np.max(np.maximum(0.0, g[u >= 1 - 1e-9]), initial=0.0) < 1e-8


def test_no_grid_sequence_beats_the_qp_value():
    """Stellfolgen-Gitter 41³ = 68 921 Folgen (N = 3): kein Gitterwert liegt unter dem QP-Wert."""
    prob = D.build_mpc("wagen", 1.0, 3, 1.0, 1.0, "DARE")
    g = np.linspace(-1, 1, 41)
    U = np.array(list(itertools.product(g, g, g)))
    for x0 in ([10.0, 0.0], [-6.0, 2.0], [1.0, 0.5]):
        f = prob.F @ np.array(x0)
        vals = 0.5 * np.einsum("ij,jk,ik->i", U, prob.H, U) + U @ f
        u = QP.qp_active_set(prob.H, f, prob.lo, prob.hi).u
        assert vals.min() >= _value(prob.H, f, u) - 1e-12


def test_solver_against_exact_fractions_for_n2():
    """Bruch-Aufzählung (msg_exact, aus der Kostenfunktion durch Differenzen) gegen die Gleitkomma-Methode an Zufallszuständen."""
    rng = SplitMix64(18)
    P = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]
    prob = D.build_mpc("wagen", 1.0, 2, 1.0, 0.5, "Q")
    for _ in range(15):
        x0 = (Fr(int(-9 + 18 * rng.random())), Fr(int(-3 + 6 * rng.random())))
        H, f, _ = hessian_by_differences(2, x0, Fr(1), P)
        u_ex, _ = E.box_qp_enum(H, f, Fr(-1, 2), Fr(1, 2))
        u = QP.qp_active_set(prob.H, prob.F @ np.array([float(x0[0]), float(x0[1])]), prob.lo, prob.hi).u
        assert np.max(np.abs(u - np.array([float(v) for v in u_ex]))) < 1e-12


def test_condensed_cost_equals_direct_sum_for_random_sequences():
    rng = SplitMix64(19)
    for N in (1, 3, 5, 8):
        prob = D.build_mpc("wagen", 1.0, N, 0.7, 1.0, "DARE")
        for _ in range(5):
            x0 = np.array([-8 + 16 * rng.random(), -3 + 6 * rng.random()])
            u = np.array([-1 + 2 * rng.random() for _ in range(N)])
            kond = QP.qp_value(u, prob.H, prob.F @ x0, float(x0 @ prob.Cmat @ x0))
            direkt = cost_direct(list(u), list(x0), 0.7, prob.P)
            assert kond == pytest.approx(direkt, rel=1e-12)


@pytest.mark.parametrize("R", [1.0, 0.1, 0.01])
def test_dare_gain_against_riccati_recursion(R):
    """P_{k+1} = Q + A'P_kA − A'P_kB (R + B'P_kB)^{-1} B'P_kA ab P_0 = Q konvergiert gegen die DARE-Lösung (200 Schritte, Abweichung höchstens 1e-9)."""
    A, B, Q = D.plant("wagen")
    P = Q.copy()
    for _ in range(200):
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    P2, K = D.dare(A, B, Q, R)
    assert np.max(np.abs(P - P2)) < 1e-9
    assert np.allclose(K, np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A, atol=1e-9)
    assert np.all(np.abs(np.linalg.eigvals(A - B @ K)) < 1)


def test_scalar_driven_cost_against_dynamic_programming_on_a_grid():
    """Wertiteration auf einem Gitter (Schritt 0,01, Stellgrenze 1, Faktor ½) gegen die gefahrenen Kosten von „LQR, dann abschneiden“ ab x = 10 (Toleranz 1e-3)."""
    A, B, Q = D.plant("skalar")
    P, K = D.dare(A, B, Q, 1.0)
    h = 0.01
    xg = np.arange(-14.0, 14.0 + 1e-9, h)
    ug = np.arange(-1.0, 1.0 + 1e-9, h)
    V = 0.5 * P[0, 0] * xg ** 2
    for _ in range(2000):
        xn = np.clip(xg[:, None] + ug[None, :], xg[0], xg[-1])
        Vn = np.interp(xn.ravel(), xg, V).reshape(xn.shape)
        Vneu = np.min(0.5 * (xg[:, None] ** 2 + ug[None, :] ** 2) + Vn, axis=1)
        d = np.max(np.abs(Vneu - V))
        V = Vneu
        if d < 1e-10:
            break
    v10 = V[int(round((10.0 - xg[0]) / h))]
    gefahren = 0.5 * D.run_loop(D.ClipLaw(K, 1.0), np.array([10.0]), A, B, Q, 1.0, T=600, tol=1e-10).cost
    mpc40 = 0.5 * D.run_loop(D.MPCLaw(D.build_mpc("skalar", 1.0, 40, 1.0, 1.0, "DARE")), np.array([10.0]), A, B, Q, 1.0, T=600, tol=1e-10).cost
    assert abs(v10 - gefahren) < 1e-3 and abs(v10 - mpc40) < 1e-3
    # MPC N = 2 mit Endkosten Q: exakte Brüche
    mpc2 = 0.5 * D.run_loop(D.MPCLaw(D.build_mpc("skalar", 1.0, 2, 1.0, 1.0, "Q")), np.array([10.0]), A, B, Q, 1.0, T=600, tol=1e-12).cost
    assert mpc2 == pytest.approx(float(E.scalar_loop(10)[0]), abs=1e-8)


def test_controllable_set_against_bounded_least_squares():
    """Steuerbar in 100 Schritten (lineares Programm) gegen den Rest der beschränkten kleinsten Quadrate min ‖Γu + A^100 x0‖ mit |u| ≤ 1 (bvls): Rest nahe 0 genau dort."""
    from scipy.optimize import lsq_linear

    A, B, _ = D.plant("wagen", 1.2)
    pts = D.grid_points((-6.0, 6.0, 1.5), (-6.0, 6.0, 1.5))
    mask = D.controllable_mask(A, B, pts)
    b = B[:, 0]
    Gam = np.column_stack([np.linalg.matrix_power(A, 99 - j) @ b for j in range(100)])
    AN = np.linalg.matrix_power(A, 100)
    rest = np.array([np.linalg.norm(lsq_linear(Gam, -AN @ np.array(p), bounds=(-1, 1), method="bvls", tol=1e-14).fun) for p in pts])
    assert np.all((rest < 1e-4) == mask)          # A^100 hat den Faktor 8·10^7: Rauschen der Reste bis 1e-6, unsteuerbare Punkte liegen bei 1e7 und mehr
    assert 0 < mask.sum() < len(pts)


def test_clip_batch_equals_the_single_loop():
    """Die vektorisierte Fassung gegen die Schleife je Startzustand (Summenreihenfolge weicht ab: relative Toleranz 1e-9)."""
    A, B, Q = D.plant("wagen")
    P, K = D.dare(A, B, Q, 1.0)
    pts = D.grid_points((-10.0, 10.0, 5.0), (-4.0, 4.0, 2.0))
    cost, end, _, steps = D.run_clip_batch(pts, K, 0.5, A, B, Q, 1.0)
    for p, c, s in zip(pts, cost, steps):
        lp = D.run_loop(D.ClipLaw(K, 0.5), np.array(p), A, B, Q, 1.0)
        assert c == pytest.approx(lp.cost, rel=1e-9) and s == lp.steps
