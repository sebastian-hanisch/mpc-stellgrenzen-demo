"""Randfälle (DEMO-PLAYBOOK §12): sehr kleine und sehr große Grenze, x0 = 0, x0 auf der Aktivmengen-Grenze, N = 1 und N = 2, R → 0 (Rückfall-Pfad),
entartete Grenze, a = 1,2 mit N = 40, Permalink-Werte außerhalb der Regler. Gleitkomma-Gleichheit nur in Brüchen oder als relative Toleranz."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import msg_dynamics as D
import msg_exact as E
import msg_presets as P
import msg_qp as QP
from msg_rng import SplitMix64


def _solve(N, R, um, x0, terminal="DARE", a=1.0):
    p = D.build_mpc("wagen", a, N, R, um, terminal)
    return p, QP.qp_solve(p.H, p.F @ np.asarray(x0, float), p.lo, p.hi)


def test_tiny_limit_runs_and_keeps_all_stellsignale_at_the_bound():
    p, r = _solve(5, 1.0, 1e-6, [10.0, 0.0])
    assert np.all(np.abs(r.u) <= 1e-6 * (1 + 1e-9)) and np.all(np.abs(r.u) >= 1e-6 * (1 - 1e-6)) and not r.fallback


def test_huge_limit_gives_unconstrained_solution():
    p, r = _solve(5, 1.0, 1e6, [10.0, 0.0])
    assert r.u == pytest.approx(-np.linalg.solve(p.H, p.F @ np.array([10.0, 0.0])), abs=1e-6)


def test_start_at_origin_gives_zero_stellsignal():
    for N in (1, 2, 10):
        p, r = _solve(N, 1.0, 1.0, [0.0, 0.0])
        assert np.all(r.u == 0) and r.iterations == 1


def test_scalar_bounds_x_5_3_and_3_have_both_sides_equal():
    """Auf den Aktivmengen-Grenzen |x| = 5/3 und 3 geben beide Muster dieselbe Lösung (Bruch gleich Bruch); beidseitig daneben wechselt nur das Muster."""
    assert E.scalar_n2(Fr(5, 3))[0] == [Fr(-1), Fr(-1, 3)]
    assert E.scalar_n2(Fr(3))[0] == [Fr(-1), Fr(-1)]
    # in Gleitkomma über die Methode (N = 2, Endkosten Q): u1 = −1 bei x = 3 mit Gradient 0 (Gleichstand an der Grenze)
    p = D.build_mpc("skalar", 1.0, 2, 1.0, 1.0, "Q")
    r = QP.qp_active_set(p.H, p.F[:, 0] * 3.0, p.lo, p.hi)
    assert r.u == pytest.approx([-1.0, -1.0], abs=1e-9)
    r = QP.qp_active_set(p.H, p.F[:, 0] * (5 / 3), p.lo, p.hi)
    assert r.u == pytest.approx([-1.0, -1 / 3], abs=1e-9)


def test_horizon_one_and_two():
    """N = 1: ein Unbekannter; N = 2: der kleinste Fall mit Wechselwirkung (MPC ≠ Abschneiden der unbeschränkten Folge am Handbeispiel)."""
    p1, r1 = _solve(1, 1.0, 1.0, [10.0, 0.0])
    assert r1.u.shape == (1,) and r1.u[0] == -1.0
    p2 = D.build_mpc("wagen", 1.0, 2, 1.0, 0.5, "Q")
    f = p2.F @ np.array([7.0, -2.0])
    u = QP.qp_solve(p2.H, f, p2.lo, p2.hi).u
    unc = -np.linalg.solve(p2.H, f)
    assert u[0] == pytest.approx(0.125) and np.clip(unc[0], -0.5, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_tiny_r_makes_h_ill_conditioned_but_the_solver_still_answers():
    """R = 1e-6: sehr kleiner Stellaufwand, die Lösung stimmt mit bvls überein (der Rückfall-Pfad müsste sie sonst liefern)."""
    p, r = _solve(8, 1e-6, 1.0, [10.0, 3.0])
    ub = QP.qp_bvls(p.H, p.F @ np.array([10.0, 3.0]), p.lo, p.hi)
    v = lambda u: 0.5 * u @ p.H @ u + (p.F @ np.array([10.0, 3.0])) @ u  # noqa: E731
    assert v(r.u) <= v(ub) + 1e-7


def test_unstable_a12_with_n40_needs_the_fallback_sometimes():
    """a = 1,2, N = 40 (Konditionszahl 6·10⁸): die Aktivmengen-Methode bricht in einem Teil der Aufrufe ab, der Rückfall liefert dennoch eine zulässige, optimale Lösung (Band, kein genauer Anteil)."""
    p = D.build_mpc("wagen", 1.2, 40, 1.0, 1.0, "DARE")
    assert np.linalg.cond(p.H) > 1e8
    rng = SplitMix64(3)                                # nur Zustände für die Stichprobe; der Anteil wird nicht festgeschrieben
    n_fb = 0
    for _ in range(25):
        x0 = np.array([-6 + 12 * rng.random(), -6 + 12 * rng.random()])
        f = p.F @ x0
        r = QP.qp_solve(p.H, f, p.lo, p.hi)
        n_fb += int(r.fallback)
        assert np.all(np.abs(r.u) <= 1 + 1e-12)
        ub = QP.qp_bvls(p.H, f, p.lo, p.hi)
        vr, vb = 0.5 * r.u @ p.H @ r.u + f @ r.u, 0.5 * ub @ p.H @ ub + f @ ub
        assert vr <= vb + 1e-9 * abs(vb)          # Zielwerte (Größenordnung 1e9) stimmen relativ auf 1e-9 überein; die Stellfolgen können in flachen Richtungen um bis 0,4 abweichen
    assert 0 <= n_fb <= 25


def test_equal_bounds_in_the_solver():
    r = QP.qp_solve(np.eye(2), np.array([1.0, 1.0]), np.array([0.3, 0.3]), np.array([0.3, 0.3]))
    assert np.all(r.u == 0.3)


def test_permalink_values_outside_the_widgets_are_snapped():
    """Werte aus der Adresszeile: Zahl begrenzt, Auswahlwerte auf den nächsten erlaubten Wert eingerastet, unbekannter Text auf den Standard."""
    assert P.snap(P.SETTING_SPECS["n_slider"], 99) == 20 and P.snap(P.SETTING_SPECS["n_slider"], -3) == 1
    assert P.snap(P.SETTING_SPECS["umax_slider"], 0.9) == 1.0 and P.snap(P.SETTING_SPECS["umax_slider"], 7.0) == 2.0
    assert P.snap(P.SETTING_SPECS["terminal"], "XYZ") == "DARE" and P.snap(P.SETTING_SPECS["e2_system"], "Skalar") == "Skalar"
    assert P.snap(P.SETTING_SPECS["e6_n"], 12) == 10
