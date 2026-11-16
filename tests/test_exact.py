"""Exakte Handrechnung (msg_exact.py) gegen die Handwerte: Skalar (Rawlings Bsp. 2.5, mit Faktor ½) und Wagen N = 2. Alles mit Brüchen, Bruch gleich Bruch."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import msg_exact as E
import msg_qp as QP
import msg_dynamics as D
from conftest import cost_direct, enum_box_qp, hessian_by_differences


def test_scalar_hessian_from_the_cost_by_hand():
    """V = ½[x² + u0² + (x+u0)² + u1² + (x+u0+u1)²] = ½u'Hu + (2x; x)'u + (3/2)x² mit H = [[3,1],[1,2]] (von Hand, auch aus den Differenzen der Kostenfunktion)."""
    # Kosten ohne Faktor ½ (Q = R = P = 1): J = 2 V; Differenzen liefern 2H und 2f
    x0 = (Fr(1),)
    H, f, J0 = hessian_by_differences(2, x0, Fr(1), [[Fr(1)]], A=((1,),), B=(1,))
    assert [[v / 2 for v in row] for row in H] == [[3, 1], [1, 2]]
    assert [v / 2 for v in f] == [2, 1]
    assert J0 / 2 == Fr(3, 2)                            # V(u = 0) = ½[x² + x² + x²] = (3/2)x² bei x = 1 (x0² + x1² + x2² = 3, halbiert)


def test_scalar_h_inverse_and_unconstrained_minimizer():
    H = np.array([[3, 1], [1, 2]])
    Hinv = np.array([[2, -1], [-1, 3]]) / 5
    assert np.allclose(H @ Hinv, np.eye(2))
    assert np.allclose(-Hinv @ np.array([2.0, 1.0]), [-0.6, -0.2])      # u = −(3/5, 1/5)·x bei x = 1


@pytest.mark.parametrize("x, u, v", [(Fr(1), (Fr(-3, 5), Fr(-1, 5)), Fr(4, 5)), (Fr(2), (Fr(-1), Fr(-1, 2)), Fr(13, 4)), (Fr(3), (Fr(-1), Fr(-1)), Fr(8)),
                                      (Fr(10), (Fr(-1), Fr(-1)), Fr(247, 2)), (Fr(5, 3), (Fr(-1), Fr(-1, 3)), Fr(20, 9)), (Fr(-2), (Fr(1), Fr(1, 2)), Fr(13, 4))])
def test_scalar_qp_values_by_hand(x, u, v):
    """V(1) = 4/5, V(2) = 13/4, V(3) = 8, V(10) = 247/2 = 123,5 (die Quelle nennt nach Quellenbericht 124, im Original nicht gelesen; ½(100 + 1 + 81 + 1 + 64))."""
    uu, vv, _ = E.scalar_n2(x)
    assert tuple(uu) == u and vv == v


def test_scalar_ten_by_hand_sum():
    assert Fr(1, 2) * (100 + 1 + 81 + 1 + 64) == Fr(247, 2)


def test_scalar_law_is_minus_sat_of_three_fifths_x_at_161_start_values():
    for k in range(-80, 81):
        x = Fr(k, 10)
        assert E.scalar_n2(x)[0][0] == -E.sat(Fr(3, 5) * x), x


def test_scalar_regions_have_the_hand_boundaries():
    """Grenze greift für u₀ ab |x| = 5/3 und für u₁ ab |x| = 3; an den Rändern beide Seiten gleich."""
    assert E.scalar_n2(Fr(5, 3))[2] == ["unten", "frei"]
    assert E.scalar_n2(Fr(3))[2] == ["unten", "unten"]
    assert E.scalar_n2(Fr(8, 5))[2] == ["frei", "frei"]            # 8/5 < 5/3
    assert E.scalar_n2(Fr(5, 3) + Fr(1, 1000))[2] == ["unten", "frei"]
    assert E.scalar_n2(Fr(3) - Fr(1, 1000))[2] == ["unten", "frei"]


def test_scalar_closed_loop_cost_is_8287_over_42():
    """x = 10, 9, …, 2 mit u = −1 (Kosten ½ Σ (x² + 1) = 393/2), ab x = 1 der Faktor 2/5 mit Rest 17/21: zusammen 8287/42 = 197,3095."""
    g, folge = E.scalar_loop(10)
    assert folge == [Fr(v) for v in range(10, 0, -1)]
    assert Fr(1, 2) * sum(v * v + 1 for v in range(2, 11)) == Fr(393, 2)
    assert Fr(17, 25) / (1 - Fr(4, 25)) == Fr(17, 21)
    assert g == Fr(393, 2) + Fr(17, 21) == Fr(8287, 42)
    assert float(g) == pytest.approx(197.3095238, abs=1e-6)


def test_scalar_closed_loop_cost_against_a_long_partial_sum():
    """Gegenprobe der geschlossenen Formel: Partialsumme über 80 Schritte mit exakten Brüchen, Rest unter 1e-12."""
    x, s = Fr(10), Fr(0)
    for _ in range(80):
        u = -E.sat(Fr(3, 5) * x)
        s += Fr(1, 2) * (x * x + u * u)
        x += u
    assert abs(float(E.scalar_loop(10)[0] - s)) < 1e-12


def test_lqr_then_clip_in_the_scalar_costs_393_over_2_plus_phi_over_2():
    """DARE P² − P − 1 = 0, K = P/(1+P) = 1/φ; gleiche Phase 1 (u = −1 solange Kx ≥ 1), dann Faktor 1/φ²; Rest ½(1 + K²)/(1 − (1−K)²) = φ/2."""
    phi = (1 + 5 ** 0.5) / 2
    A, B, Q = D.plant("skalar")
    P, K = D.dare(A, B, Q, 1.0)
    assert P[0, 0] == pytest.approx(phi, abs=1e-12) and K[0, 0] == pytest.approx(1 / phi, abs=1e-12)
    cost = 0.5 * D.run_loop(D.ClipLaw(K, 1.0), np.array([10.0]), A, B, Q, 1.0, T=600, tol=1e-12).cost
    assert cost == pytest.approx(393 / 2 + phi / 2, abs=1e-9)
    assert 0.0 < float(E.scalar_loop(10)[0]) - cost < 1e-3                  # MPC N = 2 mit Gewinn 3/5 ist minimal teurer als 1/φ


def test_wagen_hand_example_exact():
    w = E.wagen_hand()
    assert w["H"] == [[8, 2], [2, 4]] and w["f"] == [-2, -4]
    assert w["unbeschraenkt"] == [0, 1]
    assert w["mpc"] == [Fr(1, 8), Fr(1, 2)]
    assert w["abgeschnitten"] == [0, Fr(1, 2)]
    assert (w["J_unbeschraenkt"], w["J_mpc"], w["J_abgeschnitten"]) == (93, Fr(1495, 16), Fr(187, 2))
    assert w["gradient"] == [0, Fr(-7, 4)]
    assert w["J_abgeschnitten"] - w["J_mpc"] == Fr(1, 2) * 8 * (Fr(1, 8)) ** 2          # ½·H₀₀·(u₀ − u₀*)² = 1/16


def test_wagen_hand_costs_match_the_direct_sum():
    P = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]
    for u, J in (((Fr(1, 8), Fr(1, 2)), Fr(1495, 16)), ((Fr(0), Fr(1, 2)), Fr(187, 2)), ((Fr(0), Fr(1)), Fr(93))):
        assert cost_direct(list(u), (Fr(7), Fr(-2)), Fr(1), P) == J


def test_wagen_hand_grid_of_sequences_has_the_same_best_point():
    """Gitter 17 × 17 mit Schritt 1/16 in der Box: bester Punkt (1/8, 1/2) mit demselben Wert."""
    P = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]
    gitter = [Fr(k, 16) for k in range(-8, 9)]
    best = min(((cost_direct([a, b], (Fr(7), Fr(-2)), Fr(1), P), (a, b)) for a in gitter for b in gitter), key=lambda t: t[0])
    assert best == (Fr(1495, 16), (Fr(1, 8), Fr(1, 2)))


def test_three_regulation_steps_by_hand():
    """x = (7; −2) → u₀ = 1/8, x = (5; −15/8) → u₀ = 1/2, x = (25/8; −11/8) → u₀ = 27/56 (alle 9 Muster je Schritt, Brüche)."""
    P = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]
    x = (Fr(7), Fr(-2))
    erwartet = [Fr(1, 8), Fr(1, 2), Fr(27, 56)]
    for k in range(3):
        H, f, _ = hessian_by_differences(2, x, Fr(1), P)
        u, _ = E.box_qp_enum(H, f, Fr(-1, 2), Fr(1, 2))
        assert u[0] == erwartet[k]
        x = (x[0] + x[1], x[1] + u[0])


@pytest.mark.parametrize("P", [[[Fr(1), Fr(0)], [Fr(0), Fr(1)]], [[Fr(3), Fr(1)], [Fr(1), Fr(2)]]])
def test_n1_qp_equals_clipped_unconstrained_solution_exactly(P):
    """N = 1: ein Unbekannter, QP = abgeschnittene unbeschränkte Lösung, Bruch gleich Bruch an 117 ganzzahligen Zuständen (für zwei verschiedene P)."""
    ok = 0
    for a in range(-6, 7):
        for b in range(-4, 5):
            H, f, _ = hessian_by_differences(1, (Fr(a), Fr(b)), Fr(1), P)
            u, _ = E.box_qp_enum(H, f, Fr(-1, 2), Fr(1, 2))
            ok += int(u[0] == max(Fr(-1, 2), min(Fr(1, 2), -f[0] / H[0][0])))
    assert ok == 117


def test_exact_and_float_solvers_agree_on_the_hand_case():
    H, f, _ = hessian_by_differences(2, (Fr(7), Fr(-2)), Fr(1), [[Fr(1), Fr(0)], [Fr(0), Fr(1)]])
    u_float, _ = enum_box_qp([[float(v) for v in r] for r in H], [float(v) for v in f], [-0.5, -0.5], [0.5, 0.5])
    r = QP.qp_active_set(np.array(H, float), np.array(f, float), -0.5 * np.ones(2), 0.5 * np.ones(2))
    assert np.allclose(u_float, r.u, atol=1e-12)
