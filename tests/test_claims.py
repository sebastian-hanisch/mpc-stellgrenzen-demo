"""Claims: jede Zahl der README steht hier mit ihrer Bedingung und Zählung (Tabelle „Zahl → Test“ in der README).

Zwei Arten von Tests:
* **vorgerechnet** (`data/msg_frozen_*.json`, erzeugt von `tools/make_frozen.py`): die Zahl der README gleicht der gespeicherten Messung (exakt, es sind gespeicherte Werte);
* **neu gerechnet** (live): die Rechnung läuft im Test erneut und wird als Band verglichen (Zählungen ± 2, Anteile ± 0,01, Gleitkomma relativ 1e-6 bis 1e-9, Iterationszahlen ± 20 %),
  weil Gleichstände in der Aktivmenge und BLAS-Summenreihenfolgen unter Linux das letzte Bit ändern können (Lehre 2026-11-15). Gleichstände an Schwellen: die Zählung „gleich“ nutzt die
  Schwelle 1e-9 (zwischen 1e-12 und 1e-6 liegt in der Messung kein Punkt), nur Brüche werden exakt verglichen."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import msg_constants as C
import msg_dynamics as D
import msg_evaluation as ev
import msg_exact as E
from msg_rng import SplitMix64

FZ = ev.frozen()
PTS = D.grid_points(C.GRID_X1, C.GRID_X2)


def _sc(key):
    return FZ["d1"]["szenarien"][key]


def test_all_frozen_files_exist():
    assert {"kurz", "d1", "skalar", "gegenprobe", "weit", "instabil", "orakel", "rueckfall"} <= set(FZ)


# ---------------------------------------------------------------------------------------------------------------- B1 bis B3: Skalar
def test_b1_scalar_exact_values():
    """B1: u₀ = −sat(3x/5) an 161 von 161 Bruch-Startwerten, V_N⁰(10) = 247/2 = 123,5, gefahren 8287/42 = 197,3095, Voraussage 37 % unter den gefahrenen Kosten (Brüche)."""
    assert sum(E.scalar_n2(Fr(k, 10))[0][0] == -E.sat(Fr(3, 5) * Fr(k, 10)) for k in range(-80, 81)) == 161
    h = ev.hand(Fr(10))
    assert h["V"] == Fr(247, 2) and h["gefahren"] == Fr(8287, 42)
    assert round(100 * float(h["gefahren"] - h["V"]) / float(h["gefahren"])) == 37
    assert float(h["gefahren"]) == pytest.approx(197.3095, abs=5e-5) and float(h["V"]) == 123.5


def test_b2_scalar_clip_equals_mpc_frozen():
    """B2 (vorgerechnet): erstes Stellsignal MPC gegen abgeschnittenes unbeschränktes Gesetz, N = 1 bis 8, Endkosten Q und DARE: 0 Abweichungen über 1e-9 auf 2 401 Punkten; zweites Gitter (Schritt 0,013, Grenze 0,7) ebenso."""
    s3 = FZ["skalar"]["s3"]
    assert len(s3) == 16 and all(v["punkte"] == 2401 and v["abweichend"] == 0 for v in s3.values())
    assert max(v["groesste"] for v in s3.values()) < 1e-12
    s3v = FZ["skalar"]["s3_versetzt"]
    assert len(s3v) == 8 and all(v["punkte"] == 1847 and v["abweichend"] == 0 for v in s3v.values())
    assert s3["Q|2"]["gewinn"] == pytest.approx(0.6) and s3["DARE|2"]["gewinn"] == pytest.approx(0.618034, abs=1e-6)


@pytest.mark.parametrize("term, N", [("Q", 1), ("Q", 2), ("DARE", 8), ("Q", 8)])
def test_b2_scalar_clip_equals_mpc_live(term, N):
    """B2 (neu gerechnet): dieselbe Frage live; Bedingung: Grenze 1, R = 1, Gitter −12 bis 12 Schritt 0,01."""
    s = ev.Settings(n=N, terminal=term)
    r = ev.skalar_firstu(s)
    assert r["punkte"] == 2401 and r["abweichend"] == 0 and r["groesste"] < 1e-9


def test_b2_scalar_driven_costs():
    """B2: gefahrene Kosten ab x = 10 (Faktor ½): MPC N = 2 197,3095, LQR-abgeschnitten und MPC N = 40 197,3090, DP-Gitter 197,3090; MPC N = 2 liegt 0,0003 % über dem Optimum."""
    s4 = FZ["skalar"]["s4"]
    assert s4["mpc_n2"] == pytest.approx(197.3095, abs=5e-5) and s4["clip"] == pytest.approx(197.3090, abs=5e-5) and s4["mpc_n40"] == pytest.approx(197.3090, abs=5e-5)
    assert s4["dp_gitter"] == pytest.approx(197.3090, abs=5e-5)
    assert 100 * (s4["mpc_n2"] / s4["dp_gitter"] - 1) == pytest.approx(0.0003, abs=1e-4)
    r = ev.skalar_firstu(ev.Settings(n=2, terminal="Q"))                      # live, Band relativ 1e-6
    assert r["c_mpc"] == pytest.approx(s4["mpc_n2"], rel=1e-6) and r["c_clip"] == pytest.approx(s4["clip"], rel=1e-6)


# ---------------------------------------------------------------------------------------------------------------- B3 bis B6: Wagen
def test_b3_wagen_n1_equals_clip_frozen_and_live():
    """B3: N = 1: gleiche Kosten an 356 von 356 Startzuständen (u_max = 1, R = 1; vorgerechnet und neu gerechnet), Brüche 117 von 117 in tests/test_exact.py."""
    e = _sc("1|1")["mpc"]["1"]
    assert (e["billiger"], e["gleich"], e["teurer"]) == (0, 356, 0)
    g = ev.wagen_grid(ev.Settings(n=1))
    assert ev.wagen_counts(g)["gleich"] == 356


@pytest.mark.parametrize("key, billiger, gleich, vorteil", [("1|1", 146, 210, 3.09), ("1|0.5", 294, 62, 59.41), ("0.1|1", 234, 122, 16.05)])
def test_b5_cost_counts_n10_frozen(key, billiger, gleich, vorteil):
    """B5 (vorgerechnet): MPC N = 10 gegen Abschneiden auf 356 Startzuständen: billiger, gleich, teurer 0; größter Vorteil (u_max = 1 / 0,5 mit R = 1, u_max = 1 mit R = 0,1)."""
    e = _sc(key)["mpc"]["10"]
    assert (e["billiger"], e["gleich"], e["teurer"]) == (billiger, gleich, 0) and e["vorteil_max"] == pytest.approx(vorteil, abs=0.005)


@pytest.mark.parametrize("r, um, billiger, gleich", [(1.0, 1.0, 146, 210), (1.0, 0.5, 294, 62), (0.1, 1.0, 234, 122)])
def test_b5_cost_counts_n10_live_band(r, um, billiger, gleich):
    """B5 (neu gerechnet, Band ± 2): dieselben Zählungen; teurer bleibt 0 (Schwelle 1e-9)."""
    c = ev.wagen_counts(ev.wagen_grid(ev.Settings(n=10, umax=um, r=r)))
    assert abs(c["billiger"] - billiger) <= 2 and abs(c["gleich"] - gleich) <= 2 and c["teurer"] == 0 and c["verglichen"] == 356


@pytest.mark.parametrize("n, r, um, billiger", [(2, 1.0, 1.0, 142), (2, 1.0, 2.0, 0), (2, 10.0, 1.0, 0), (2, 10.0, 0.75, 30), (10, 1.0, 2.0, 0), (10, 10.0, 1.0, 0), (10, 10.0, 0.75, 30)])
def test_b5_better_than_clipping_needs_an_active_bound_on_the_grid(n, r, um, billiger):
    """B5: „ab N = 2 billiger“ gilt nur, wenn die Grenze auf dem Gitter greift: u_max = 1, R = 1: 142 bei N = 2; u_max = 2 (R = 1) und R = 10 mit u_max = 1: 0 von 356; R = 10, u_max = 0,75: 30 (neu gerechnet, Band ± 2)."""
    c = ev.wagen_counts(ev.wagen_grid(ev.Settings(n=n, umax=um, r=r)))
    assert abs(c["billiger"] - billiger) <= 2 and c["teurer"] == 0 and c["verglichen"] == 356


def test_b5_threshold_does_not_decide_the_count():
    """B5: zwischen relativ 1e-12 und 1e-6 liegt kein Punkt; der kleinste Unterschied darüber ist 2,5·10⁻⁶ (u_max = 1) bzw. 1,8·10⁻⁵ (u_max = 0,5); (vorgerechnet)."""
    sw = FZ["d1"]["schwelle"]
    assert sw["1|1"]["dazwischen"] == 0 and sw["1|0.5"]["dazwischen"] == 0
    assert sw["1|1"]["kleinster_ueber_1e-12"] == pytest.approx(2.5e-6, rel=0.05) and sw["1|0.5"]["kleinster_ueber_1e-12"] == pytest.approx(1.8e-5, rel=0.05)
    assert sw["1|1"]["stufen"]["1e-12"]["clip_teurer"] == sw["1|1"]["stufen"]["1e-06"]["clip_teurer"] == 146


@pytest.mark.parametrize("key, median, p90, mx", [("1|1", 0.00, 0.35, 3.19), ("1|0.5", 1.43, 10.17, 146.37), ("0.1|1", 0.08, 3.34, 19.11)])
def test_b5_clip_surcharge_against_reference(key, median, p90, mx):
    """B5: Aufschlag von „LQR, dann abschneiden“ gegen die Referenz (MPC N = 60, DARE): Median / 90-%-Quantil / Maximum in % (vorgerechnet)."""
    c = _sc(key)["clip"]
    assert c["konvergiert"] == 356 and c["median"] == pytest.approx(median, abs=0.005) and c["p90"] == pytest.approx(p90, abs=0.005) and c["max"] == pytest.approx(mx, abs=0.005)


def test_b5_clip_surcharge_live_against_frozen_reference():
    """B5 (neu gerechnet): Aufschlag des Abschneidens gegen die vorgerechnete Referenz, u_max = 0,5, R = 1: Maximum 146,37 % (Band ± 0,05), Median 1,43 % (± 0,05)."""
    s = ev.Settings(umax=0.5)
    g = ev.wagen_grid(s)
    ref = np.array(FZ["d1"]["ref_costs"][ev.ref_key(1.0, 0.5)])
    auf = 100 * (g.cost_clip / ref - 1)
    assert auf.max() == pytest.approx(146.37, abs=0.05) and np.median(auf) == pytest.approx(1.43, abs=0.05)


def test_b5_reference_costs_recomputed_at_a_few_points():
    """Die vorgerechnete Referenz (N = 60) gegen eine neue Rechnung an 6 Startzuständen (relativ 1e-6)."""
    idx = [0, 57, 120, 200, 301, 355]
    for key, (r, um) in (("1|1", (1.0, 1.0)), ("0.1|0.25", (0.1, 0.25))):
        live = D.reference_costs(r, um, [PTS[i] for i in idx])
        assert live == pytest.approx(np.array(FZ["d1"]["ref_costs"][key])[idx], rel=1e-6)


def test_b6_random_instances_frozen():
    """B6 (vorgerechnet): 1 000 Zufallsinstanzen (R ∈ {0,1; 1; 10}, u_max ∈ [0,2; 2], x₀ ∈ [−30, 30] × [−12, 12], Zufalls-Seed 14, SplitMix64, 3 000 Schritte): MPC mit N = 2, 3, 5 in 0 von 1 000 teurer."""
    g = FZ["gegenprobe"]
    assert g["instanzen"] == 1000 and g["clip_konvergiert"] == 1000 and g["seed"] == 14 and g["schritte"] == 3000
    for N, (gl, bi) in {"2": (232, 768), "3": (193, 807), "5": (193, 807)}.items():
        e = g["n"][N]
        assert e["teurer"] == 0 and e["gleich"] == gl and e["billiger"] == bi and e["gleich"] + e["billiger"] == 1000
    assert [round(g["n"][N]["vorteil_max_prozent"], 1) for N in ("2", "3", "5")] == [72.7, 74.9, 88.2]


def test_b6_random_instances_live_first_200():
    """B6 (neu gerechnet): die ersten 200 Instanzen desselben SplitMix64-Stroms: teurer in 0 von 200 (Schwelle 1e-9), MPC konvergiert überall."""
    rng = SplitMix64(14)
    teurer = 0
    for _ in range(200):
        R = (0.1, 1.0, 10.0)[rng.below(3)]
        um = 0.2 + 1.8 * rng.random()
        x0 = np.array([-30.0 + 60.0 * rng.random(), -12.0 + 24.0 * rng.random()])
        p = D.build_mpc("wagen", 1.0, 3, R, um, "DARE")
        cl = D.run_loop(D.ClipLaw(p.K, um), x0, p.A, p.B, p.Q, R, T=3000)
        cm = D.run_loop(D.MPCLaw(p), x0, p.A, p.B, p.Q, R, T=3000)
        assert cm.end_norm < 1e-6
        teurer += int(cm.cost > cl.cost * (1 + 1e-9))
    assert teurer == 0


def test_b7_scaling():
    """B7: Kosten von (c·x₀, c·u_max) sind c² mal die von (x₀, u_max) (Abweichung höchstens 1e-12; vorgerechnet 2,2·10⁻¹⁶, neu gerechnet mit Band 1e-9)."""
    assert FZ["kurz"]["d2_skalierung_max_abw"] < 1e-12
    sc = ev.scaling_check(ev.Settings(x1=4.0, x2=1.0))
    assert sc["max_abw"] < 1e-9 and len(sc["faktoren"]) == 5


def test_b7_same_grid_scaled_gives_the_same_surcharge_map():
    """B7: u_max = 0,5 auf [−10, 10] × [−4, 4] ist u_max = 1 auf [−20, 20] × [−8, 8]: gleiche relative Mehrkosten (neu gerechnet, 20 Startzustände)."""
    rng = SplitMix64(41)
    p_a = D.build_mpc("wagen", 1.0, 10, 1.0, 0.5, "DARE")
    p_b = D.build_mpc("wagen", 1.0, 10, 1.0, 1.0, "DARE")
    for _ in range(20):
        x0 = np.array([-10 + 20 * rng.random(), -4 + 8 * rng.random()])
        a = D.run_loop(D.ClipLaw(p_a.K, 0.5), x0, p_a.A, p_a.B, p_a.Q, 1.0).cost / D.run_loop(D.MPCLaw(p_a), x0, p_a.A, p_a.B, p_a.Q, 1.0).cost
        b = D.run_loop(D.ClipLaw(p_b.K, 1.0), 2 * x0, p_b.A, p_b.B, p_b.Q, 1.0).cost / D.run_loop(D.MPCLaw(p_b), 2 * x0, p_b.A, p_b.B, p_b.Q, 1.0).cost
        assert a == pytest.approx(b, rel=1e-9)


def test_b8_horizon_tables_frozen():
    """B8 (vorgerechnet): größter Aufschlag gegen die Referenz je Horizont (Endkosten DARE, 356 Startzustände, %)."""
    f = lambda key, n: _sc(key)["mpc"][str(n)]["max"]  # noqa: E731
    assert [round(f("1|1", n), 2) for n in (1, 2, 3)] == [3.19, 0.70, 0.05] and all(f("1|1", n) < 0.005 for n in (4, 5, 7, 10, 20))
    assert [round(f("1|0.5", n), 2) for n in (1, 2, 3, 4, 5)] == [146.37, 34.77, 7.78, 1.50, 0.59] and round(f("1|0.5", 7), 2) == 0.02
    assert all(f("1|0.5", n) < 0.005 for n in (10, 20))
    assert [round(f("0.1|1", n), 2) for n in (1, 2, 3)] == [19.11, 2.37, 0.24] and round(f("0.1|1", 4), 4) == 0.0053
    assert all(f("0.1|1", n) < 0.005 for n in (5, 7, 10, 20))


@pytest.mark.parametrize("key, n, erwartet", [("1|1", 2, 0.70), ("1|1", 3, 0.05), ("1|0.5", 4, 1.50), ("1|0.5", 7, 0.02)])
def test_b8_horizon_live_band(key, n, erwartet):
    """B8 (neu gerechnet, Band ± 0,01 Prozentpunkte): Aufschlag gegen die vorgerechnete Referenz."""
    r, um = (1.0, 1.0) if key == "1|1" else (1.0, 0.5)
    s = ev.Settings(n=n, umax=um, r=r)
    st = ev.horizon_stats(s, ev.wagen_grid(s))
    assert st["konvergiert"] == 356 and st["max"] == pytest.approx(erwartet, abs=0.01)


def test_b9_prediction_against_driven_frozen_and_live():
    """B9: x₀ = (10; 0), u_max = 1, R = 1: V₂ = 374,8781, gefahren 375,4336; ab N = 5 gleich 375,4336; x₀ = (−10; 4): V₂ = 179,5462, gefahren 179,8218 (vorgerechnet und neu gerechnet, relativ 1e-6)."""
    d3 = FZ["kurz"]["d3"]
    assert d3["10;0"]["2"]["vorhergesagt"] == pytest.approx(374.8781, abs=5e-5) and d3["10;0"]["2"]["gefahren"] == pytest.approx(375.4336, abs=5e-5)
    assert all(d3["10;0"][n]["vorhergesagt"] == pytest.approx(375.4336, abs=5e-5) for n in ("5", "10", "60"))
    assert d3["-10;4"]["2"]["vorhergesagt"] == pytest.approx(179.5462, abs=5e-5) and d3["-10;4"]["5"]["gefahren"] == pytest.approx(179.8218, abs=5e-5)
    assert d3["5;-2"]["2"]["vorhergesagt"] == pytest.approx(d3["5;-2"]["2"]["gefahren"], rel=1e-9)
    pr = ev.prediction(ev.Settings(n=2, x1=10.0, x2=0.0))
    assert pr["n"]["V"] == pytest.approx(374.8781, rel=1e-6) and pr["n"]["gefahren"] == pytest.approx(375.4336, rel=1e-6) and pr["ref"]["gefahren"] == pytest.approx(375.4336, rel=1e-6)
    assert 100 * (pr["n"]["gefahren"] / pr["n"]["V"] - 1) == pytest.approx(0.15, abs=0.01)


def test_b10_without_endcost_frozen_and_live():
    """B10: Endkosten nur Q, u_max = 1, R = 1: N = 1 konvergiert in 12 von 356 Startzuständen; N = 2 in 356, Aufschlag gegen die Referenz Median 0,73 %, Maximum 10,15 %, MPC teurer als Abschneiden in 322 von 356; N = 3: Maximum 0,26 %, teurer in 238; N = 5: Maximum 0,02 %, teurer in 210."""
    d6 = FZ["d1"]["d6"]
    assert d6["1"]["konvergiert"] == 12 and d6["2"]["konvergiert"] == 356 and d6["2"]["teurer"] == 322 and d6["3"]["teurer"] == 238 and d6["5"]["teurer"] == 210
    assert round(d6["2"]["median"], 2) == 0.73 and round(d6["2"]["max"], 2) == 10.15 and round(d6["3"]["max"], 2) == 0.26 and round(d6["5"]["max"], 2) == 0.02
    c = ev.wagen_counts(ev.wagen_grid(ev.Settings(n=2, terminal="Q")))
    assert abs(c["teurer"] - 322) <= 2 and c["verglichen"] == 356
    c1 = ev.wagen_counts(ev.wagen_grid(ev.Settings(n=1, terminal="Q")))
    assert abs(c1["mpc_konvergiert"] - 12) <= 2


def test_b4_first_gain_and_first_stellsignal():
    """B4: Mit Endkosten DARE ist der unbeschränkte erste Gewinn für N = 1 bis 10 gleich K (größte Abweichung 5,6·10⁻¹⁵ bei R = 1, 2,1·10⁻¹⁴ bei R = 0,1, vorgerechnet); mit P = Q nicht (0,74 bei N = 1,
    0,24 bei N = 2). Im Fenster x₁ ∈ [−3, 3], x₂ ∈ [−1,5; 1,5] (N = 10, u_max = 1; 325 Gitterpunkte) liefert das QP −Kx an 157 von 325 Gitterpunkten; das erste Stellsignal weicht von clip(−Kx) an 88 von 1 353 Gitterpunkten ab, größte Abweichung 0,307."""
    k = FZ["kurz"]
    assert k["d4"]["1.0"]["dare_max"] < 1e-12 and k["d4"]["0.1"]["dare_max"] < 1e-12
    assert k["d4"]["1.0"]["q"]["1"] == pytest.approx(0.7439, abs=5e-4) and k["d4"]["1.0"]["q"]["2"] == pytest.approx(0.2439, abs=5e-4)
    assert (k["d4_innen"]["gleich"], k["d4_innen"]["von"]) == (157, 325) and (k["d5"]["abweichend"], k["d5"]["punkte"]) == (88, 1353)
    assert k["d5"]["groesste"] == pytest.approx(0.307, abs=5e-4)
    live = ev.firstu_check(ev.Settings(n=10))
    assert live["punkte"] == 1353 and abs(live["abweichend"] - 88) <= 3 and live["groesste"] == pytest.approx(0.307, abs=2e-3)
    assert round(100 * 88 / 1353, 1) == 6.5


def test_b16_riccati_gain():
    """B16: DARE-Gewinn des Wagens bei R = 1: K = (0,42208; 1,24393); aus (10; 0) verlangt das Gesetz u = −4,22."""
    p = D.build_mpc("wagen", 1.0, 5, 1.0, 1.0, "DARE")
    assert p.K[0] == pytest.approx([0.42208, 1.24393], abs=5e-6) and (-p.K @ np.array([10.0, 0.0]))[0] == pytest.approx(-4.22, abs=5e-3)
    ew = np.linalg.eigvals(p.A - p.B @ p.K)
    assert sorted(ew.real) == pytest.approx([0.378, 0.378], abs=5e-4) and sorted(abs(ew.imag)) == pytest.approx([0.188, 0.188], abs=5e-4)


# ---------------------------------------------------------------------------------------------------------------- B11: instabil
def test_b11_unstable_counts_frozen():
    """B11 (vorgerechnet): Gitter −6 bis 6 (Schritt 0,5), 624 Startzustände, u_max = 1, R = 1: a = 1,2: Abschneiden 332, MPC N = 3 / 5 / 10 / 20: 378 / 414 / 462 / 494; steuerbar 494; MPC in 0 nicht steuerbaren Punkten konvergent."""
    e = FZ["instabil"]["a"]
    cnt = lambda a, k: sum(e[a][k]) if k in ("clip", "steuerbar") else {n: sum(v) for n, v in e[a]["mpc"].items()}  # noqa: E731
    assert FZ["instabil"]["punkte"] == 624
    assert cnt("1.2", "clip") == 332 and cnt("1.2", "mpc") == {"3": 378, "5": 414, "10": 462, "20": 494} and cnt("1.2", "steuerbar") == 494
    for a in e:
        for n, v in e[a]["mpc"].items():
            assert not any(m and not c for m, c in zip(v, e[a]["steuerbar"])), (a, n)
        assert not any(m and not c for m, c in zip(e[a]["clip"], e[a]["steuerbar"])), a
    assert sum(1 for m, c in zip(e["1.2"]["mpc"]["20"], e["1.2"]["steuerbar"]) if c and not m) == 0
    assert sum(1 for m, c in zip(e["1.2"]["clip"], e["1.2"]["steuerbar"]) if c and not m) == 162
    for a in e:                                                        # MPC N = 20 konvergiert überall, wo Abschneiden konvergiert
        assert not any(c and not m for c, m in zip(e[a]["clip"], e[a]["mpc"]["20"])), a
    assert all(sum(e[a][k]) == 624 for a in ("1", "1.05") for k in ("clip", "steuerbar")) and all(sum(v) == 624 for a in ("1", "1.05") for v in e[a]["mpc"].values())


def test_b11_unstable_other_a_frozen():
    """B11 (vorgerechnet), weitere a: a = 1,1: Abschneiden 540, MPC N = 3: 608, ab N = 5 alle 624; a = 1,15: steuerbar 616, Abschneiden 410, MPC N = 20: 610 (6 steuerbare verfehlt); a = 1,25: steuerbar 392, MPC N = 20: 392."""
    e = FZ["instabil"]["a"]
    assert sum(e["1.1"]["clip"]) == 540 and sum(e["1.1"]["mpc"]["3"]) == 608 and sum(e["1.1"]["mpc"]["5"]) == 624
    assert sum(e["1.15"]["steuerbar"]) == 616 and sum(e["1.15"]["clip"]) == 410 and sum(e["1.15"]["mpc"]["20"]) == 610
    assert sum(e["1.25"]["steuerbar"]) == 392 and sum(e["1.25"]["mpc"]["20"]) == 392 and sum(e["1.25"]["clip"]) == 276


def test_b11_unstable_live_subset():
    """B11 (neu gerechnet, 40 Startzustände, a = 1,2): Konvergenz von Abschneiden und MPC (N = 5) und die steuerbare Menge stimmen mit der Karte überein (höchstens 2 Abweichungen, Gleichstände am Rand des Gebiets)."""
    pts = D.grid_points(C.UNSTABLE_GRID, C.UNSTABLE_GRID)
    idx = list(range(3, 624, 16))[:40]
    sub = [pts[i] for i in idx]
    A, B, Q = D.plant("wagen", 1.2)
    P, K = D.dare(A, B, Q, 1.0)
    e = FZ["instabil"]["a"]["1.2"]
    clip = [int(D.converges(D.ClipLaw(K, 1.0), p, A, B)) for p in sub]
    mpc = [int(D.converges(D.MPCLaw(D.build_mpc("wagen", 1.2, 5, 1.0, 1.0, "DARE")), p, A, B)) for p in sub]
    ctrl = [int(v) for v in D.controllable_mask(A, B, sub)]
    assert sum(a != e["clip"][i] for a, i in zip(clip, idx)) <= 2
    assert sum(a != e["mpc"]["5"][i] for a, i in zip(mpc, idx)) <= 2
    assert sum(a != e["steuerbar"][i] for a, i in zip(ctrl, idx)) <= 2


def test_b12_large_states_frozen():
    """B12 (vorgerechnet): Abschneiden am Doppelintegrator: 440 von 440 Gitterpunkten (x₁ ±300, x₂ ±60, u_max = 1, R = 1 / 0,1 / 0,01, 3 000 Schritte) konvergieren; 500 von 500 Zufallsstarts in [−1 000, 1 000] × [−200, 200] (Zufalls-Seed 14):
    Median 3 293 Schritte bis Norm unter 1e-6, höchstens 11 326; größte Norm der Bahn im Median das 11,4-Fache, höchstens das 93,6-Fache der Startnorm. MPC (N = 10) an den ersten 40 dieser Starts: 40 konvergieren."""
    w = FZ["weit"]
    assert all(v["punkte"] == 440 and v["nicht_konvergiert"] == 0 for v in w["gitter"].values())
    z = w["zufall"]
    assert (z["starts"], z["nicht_konvergiert"], z["schritte_median"], z["schritte_max"]) == (500, 0, 3293, 11326)
    assert round(z["verhaeltnis_median"], 1) == 11.4 and round(z["verhaeltnis_max"], 1) == 93.6
    m = w["mpc"]
    assert (m["starts"], m["konvergiert"], m["N"]) == (40, 40, 10) and round(m["verhaeltnis_median"], 1) == 14.4 and round(m["verhaeltnis_max"], 1) == 64.4
    assert round(m["kosten_vs_clip_min"], 2) == 0.16 and round(m["kosten_vs_clip_max"], 2) == 0.57


def test_b12_large_states_live_subset():
    """B12 (neu gerechnet, 60 Zufallsstarts mit demselben Strom): Abschneiden konvergiert überall; Überschwingen im Median zwischen dem 8- und 15-Fachen (Band)."""
    rng = SplitMix64(14)
    st = np.array([[-1000.0 + 2000.0 * rng.random(), -200.0 + 400.0 * rng.random()] for _ in range(60)])
    A, B, Q = D.plant("wagen")
    _, K = D.dare(A, B, Q, 1.0)
    _, end, gmax, steps = D.run_clip_batch(st, K, 1.0, A, B, Q, 1.0, T=200000, tol=1e-6)
    assert np.all(end < 1e-6) and 8 < np.median(gmax / np.linalg.norm(st, axis=1)) < 15


# ---------------------------------------------------------------------------------------------------------------- B13 bis B15
def test_b13_effort_frozen():
    """B13 (vorgerechnet): u_max = 1, R = 1, Kaltstart, 10 812 QP-Aufrufe entlang der 356 Bahnen: Iterationen im Mittel 2,37 (N = 5), 2,65 (N = 10), 2,81 (N = 20), Maximum 10 / 27 / 60; in 13,9 % der Aufrufe mindestens eine Grenze aktiv; 37 / 77 / 81 Vorzeichenmuster."""
    d7 = FZ["d1"]["d7"]
    assert all(d7[n]["aufrufe"] == 10812 and d7[n]["rueckfaelle"] == 0 for n in d7)
    assert [round(d7[n]["iter_mittel"], 2) for n in ("5", "10", "20")] == [2.37, 2.65, 2.81] and [d7[n]["iter_max"] for n in ("5", "10", "20")] == [10, 27, 60]
    assert round(100 * d7["10"]["anteil_aktiv"], 1) == 13.9 and [d7[n]["muster"] for n in ("5", "10", "20")] == [37, 77, 81]


def test_b13_effort_live_band():
    """B13 (neu gerechnet, Band ± 20 % bei Mittel, Maxima und Mustern; Aufrufe ± 1 %): N = 10, Kaltstart, 356 Bahnen."""
    prob = D.build_mpc("wagen", 1.0, 10, 1.0, 1.0, "DARE")
    law = D.MPCLaw(prob, record=True)
    for x0 in PTS:
        D.run_loop(law, x0, prob.A, prob.B, prob.Q, 1.0)
    its = np.array(law.iterations)
    assert abs(law.calls - 10812) <= 108 and abs(its.mean() - 2.65) <= 0.2 * 2.65 and abs(its.max() - 27) <= 0.2 * 27 and abs(len(law.patterns) - 77) <= 0.2 * 77
    assert abs(np.mean(np.array(law.active) > 0) - 0.139) <= 0.01 and law.fallbacks == 0


def test_b14_oracle_deviations_frozen():
    """B14 (vorgerechnet): 400 Zufalls-QPs (N = 1 bis 7): eigene Methode gegen Aufzählung, bvls und SLSQP größte Abweichung unter 1e-12 / 1e-12 / 1e-6; 240 MPC-QPs unter 1e-12 / 1e-12 / 1e-4; N = 10 bis 60: bvls liegt höchstens 1e-9 über der
    eigenen Lösung (vorgerechnet 4,7·10⁻¹⁰ bei N = 60), SLSQP bei N = 60 um 7·10⁵ (versagt); größte KKT-Verletzung höchstens 1e-9; Stellfolgen-Gitter (68 921 Folgen) nie unter dem QP-Wert."""
    o = FZ["orakel"]
    z, m = o["zufalls_qps"], o["mpc_qps"]
    assert z["anzahl"] == 400 and z["abweichung"]["Aufzählung"] < 1e-12 and z["abweichung"]["bvls"] < 1e-12 and z["abweichung"]["SLSQP"] < 1e-6
    assert m["anzahl"] == 240 and m["abweichung"]["Aufzählung"] < 1e-12 and m["abweichung"]["bvls"] < 1e-12 and m["abweichung"]["SLSQP"] < 1e-4
    assert z["mit_aktiver_grenze"] == 376
    for n, v in o["gross"].items():
        assert v["wert_ueberschuss_bvls"] < 1e-9 and v["kkt_max"] < 1e-9, n
    assert o["gross"]["60"]["wert_ueberschuss_slsqp"] > 1e5 and 1e5 < o["gross"]["60"]["cond"] < 1e7
    assert o["gitter"]["folgen"] == 68921 and all(f["nie_darunter"] for f in o["gitter"]["faelle"])


def test_b15_fallback_frozen():
    """B15 (vorgerechnet): Rückfall der Aktivmengen-Methode (auf bvls): 0 bei Doppelintegrator (N = 3 bis 40, je etwa 15 000 Aufrufe) und bei a = 1,2 für N ≤ 20 (27 000 bis 32 000 Aufrufe); bei a = 1,2 und N = 40
    (Konditionszahl 6,5·10⁸) zwischen 5 und 50 % der 27 032 Aufrufe (Band, keine genaue Zahl)."""
    r = FZ["rueckfall"]
    for key, v in r.items():
        if key != "1.2|40":
            assert v["rueckfaelle"] == 0, key
    v = r["1.2|40"]
    assert v["aufrufe"] == 27032 and 0.05 * v["aufrufe"] <= v["rueckfaelle"] <= 0.5 * v["aufrufe"] and 1e8 < v["cond"] < 1e9
    assert r["1|40"]["aufrufe"] == 15020


# ---------------------------------------------------------------------------------------------------------------- B17 und Presets
def test_b17_wagen_hand_example():
    """B17: Wagen N = 2, Q = P = I, R = 1, u_max = ½, x₀ = (7; −2): unbeschränkt (0; 1), MPC (1/8; 1/2), abgeschnittene Folge (0; 1/2); Kosten 93, 1495/16 = 93,4375, 187/2 (Brüche, tests/test_exact.py)."""
    w = E.wagen_hand()
    assert w["mpc"] == [Fr(1, 8), Fr(1, 2)] and w["abgeschnitten"] == [0, Fr(1, 2)] and float(w["J_mpc"]) == 93.4375


def _verdicts(s: ev.Settings):
    """Die Kernfrage-Verdikte, wie die App sie für diese Einstellungen zeigt (Wagen: Zählung über 356 Startzustände; Skalar: erstes Stellsignal)."""
    grid = ev.wagen_grid(s)
    counts = ev.wagen_counts(grid)
    if s.e2_system == "Skalar":
        return ev.verdict_skalar(ev.skalar_firstu(s), s), counts
    return ev.verdict_wagen(counts, s, ev.firstu_check(s)), counts


def test_preset_default_reports_advantage_with_counts():
    (kind, text), _ = _verdicts(ev.Settings(**C.PRESETS["Wagen, Grenze 1 (Voreinstellung)"]))
    assert kind == "success" and "von 356 Startzuständen billiger" in text and "größter Vorteil 3,09 %" in text


def test_preset_scalar_hand_and_scalar_clip_report_no_advantage():
    """Die Skalar-Presets dürfen keinen Vorteil von MPC melden (gleich in 2 401 von 2 401)."""
    for name in ("Von Hand: Skalar, x = 10", "Abschneiden = MPC im Skalar"):
        s = ev.Settings(**C.PRESETS[name])
        (kind, text), _ = _verdicts(s)
        assert kind == "info" and "2 401 von 2 401" in text and "bringt das QP gegenüber dem Abschneiden nichts" in text and "billiger" not in text, name


def test_preset_narrow_limit_reports_the_larger_advantage():
    (kind, text), c = _verdicts(ev.Settings(**C.PRESETS["Enge Grenze 0,5"]))
    assert kind == "success" and "größter Vorteil 59,41 %" in text and abs(c["billiger"] - 294) <= 2


def test_preset_short_horizon_without_endcost_warns():
    (kind, text), c = _verdicts(ev.Settings(**C.PRESETS["Zu kurzer Horizont ohne Endkosten"]))
    assert kind == "warning" and "**teurer**" in text and abs(c["teurer"] - 322) <= 2


def test_preset_unstable_reports_the_counts_of_the_map():
    s = ev.Settings(**C.PRESETS["Instabile Strecke a = 1,2"])
    kind, text = ev.verdict_instabil(ev.instabil_maps(s.e6_a, s.e6_n), s.e6_a, s.e6_n)
    assert kind == "success" and "Abschneiden konvergiert in 332, MPC in 494" in text and "494 steuerbar" in text


def test_a_preset_without_effect_reports_no_effect():
    """Ohne Grenze, die greift (u_max = 2, kleiner Start): kein Preset-Effekt wird gemeldet. Start (1; 0), N = 10: beide Regeln kosten dasselbe."""
    s = ev.Settings(umax=2.0, x1=1.0, x2=0.0)
    tr = ev.trajectory(s)
    kind, text = ev.verdict_start(tr, s)
    assert kind == "info" and "dasselbe" in text
    assert D.compare_counts([tr["clip"].cost], [tr["mpc"].cost])[:3] == (0, 1, 0)


def test_limits_table_numbers_come_from_the_frozen_files():
    rows = ev.limits_rows()
    text = " ".join(r[1] for r in rows)
    assert "5544 von 27 032 Aufrufen" in text
    assert "11,4-Fache" in text and "93,6-Fache" in text and "500 Zufallsstarts" in text
