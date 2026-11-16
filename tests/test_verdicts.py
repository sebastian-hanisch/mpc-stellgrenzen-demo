"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text. Ein Verdikt ist eine reine Funktion (Art, Text)."""

from fractions import Fraction as Fr

import numpy as np

import msg_dynamics as D
import msg_evaluation as ev

S = ev.Settings()


def test_formats():
    assert ev.de(3.14159, 2) == "3,14" and ev.zahl(1353) == "1 353" and ev.zahl(356) == "356"
    assert ev.sci(0.0) == "0" and ev.sci(1.581e-13) == "1,6e-13"
    assert ev.frac_text(Fr(-1, 3)) == "−1/3" and ev.frac_text(Fr(2)) == "2"


def test_verdict_wagen_success_names_every_count():
    c = {"punkte": 356, "verglichen": 356, "mpc_konvergiert": 356, "clip_konvergiert": 356, "billiger": 146, "gleich": 210, "teurer": 0, "vorteil_max": 3.088, "nachteil_max": 0.0}
    kind, text = ev.verdict_wagen(c, S)
    assert kind == "success"
    assert "in 146 von 356 Startzuständen billiger" in text and "gleich in 210" in text and "teurer in 0" in text and "größter Vorteil 3,09 %" in text


def test_verdict_wagen_without_difference_claims_no_advantage():
    c = {"punkte": 356, "verglichen": 356, "mpc_konvergiert": 356, "clip_konvergiert": 356, "billiger": 0, "gleich": 356, "teurer": 0, "vorteil_max": 0.0, "nachteil_max": 0.0}
    kind, text = ev.verdict_wagen(c, S)
    assert kind == "info" and "kein Unterschied" in text and "Gleiche Kosten in 356 von 356" in text and "MPC nirgends billiger" in text and "teurer" not in text


def test_verdict_wagen_warns_when_mpc_is_dearer_and_names_the_count():
    c = {"punkte": 356, "verglichen": 356, "mpc_konvergiert": 356, "clip_konvergiert": 356, "billiger": 34, "gleich": 0, "teurer": 322, "vorteil_max": 2.95, "nachteil_max": 10.15}
    kind, text = ev.verdict_wagen(c, ev.Settings(n=2, terminal="Q"))
    assert kind == "warning" and "in 322 von 356 Startzuständen **teurer**" in text and "10,15 %" in text and "billiger in 34" in text


def test_verdict_wagen_names_the_convergence_counts():
    c = {"punkte": 356, "verglichen": 12, "mpc_konvergiert": 12, "clip_konvergiert": 356, "billiger": 0, "gleich": 0, "teurer": 12, "vorteil_max": 0.0, "nachteil_max": 1.09}
    kind, text = ev.verdict_wagen(c, ev.Settings(n=1, terminal="Q"))
    assert "MPC konvergiert in 12 von 356" in text and "Abschneiden in 356" in text


def test_verdict_wagen_no_common_start():
    c = {"punkte": 356, "verglichen": 0, "mpc_konvergiert": 0, "clip_konvergiert": 356, "billiger": 0, "gleich": 0, "teurer": 0, "vorteil_max": 0.0, "nachteil_max": 0.0}
    assert ev.verdict_wagen(c, S)[0] == "warning"


def test_verdict_skalar_equal_and_different():
    r = {"abweichend": 0, "punkte": 2401, "groesste": 1.581e-13, "gewinn": 0.618034, "gewinn_dare": 0.618034, "c_mpc": 197.3095, "c_clip": 197.3090, "x0": 10.0}
    kind, text = ev.verdict_skalar(r, ev.Settings(n=8))
    assert kind == "info" and "2 401 von 2 401" in text and "1,6e-13" in text and "nichts" in text and "nicht erklärt" in text
    assert "197,3095" in text and "197,3090" in text
    r2 = dict(r, abweichend=5, groesste=0.25)
    assert "in 5 von 2 401 Punkten ab" in ev.verdict_skalar(r2, ev.Settings(n=8))[1] and "nichts" not in ev.verdict_skalar(r2, ev.Settings(n=8))[1]


def test_verdict_hand_regions_and_numbers():
    # |x| ≤ 5/3: keine Grenze
    kind, text = ev.verdict_hand(ev.hand(Fr(1)))
    assert kind == "info" and "keine Grenze greift" in text and "V = 4/5" in text and "17/21" in text
    # 5/3 ≤ |x| ≤ 3: nur u0
    assert "nur u₀ liegt an der Grenze" in ev.verdict_hand(ev.hand(Fr(2)))[1]
    # x = 10: beide; vorhergesagt 247/2 = 123,50, gefahren 8287/42 = 197,31, 37 % darunter
    t = ev.verdict_hand(ev.hand(Fr(10)))[1]
    assert "beide Stellsignale liegen an der Grenze" in t and "247/2 = 123,50" in t and "8287/42 = 197,31" in t and "**37 %**" in t
    assert ev.verdict_hand(ev.hand(Fr(0)))[0] == "info"


def test_verdict_horizon_threshold_and_counts():
    st = {"punkte": 356, "konvergiert": 356, "median": 0.0, "max": 0.7}
    pr = {"x0": [10.0, 0.0], "n": {"N": 2, "V": 374.8781, "gefahren": 375.4336}, "ref": {"N": 60, "V": 375.4336, "gefahren": 375.4336}}
    kind, text = ev.verdict_horizon(S, st, pr)
    assert kind == "info" and "Median 0,00 %" in text and "Maximum 0,70 %" in text and "374,88" in text and "375,43" in text
    assert ev.verdict_horizon(S, dict(st, konvergiert=12), pr)[0] == "warning"
    assert "In 344 Startzuständen konvergiert MPC nicht" in ev.verdict_horizon(S, dict(st, konvergiert=12), pr)[1]
    assert ev.verdict_horizon(S, dict(st, max=1.5), pr)[0] == "warning"
    assert "nicht vorgerechnet" in ev.verdict_horizon(S, None, pr)[1]
    assert "in keinem der 356" in ev.verdict_horizon(S, {"punkte": 356, "konvergiert": 0}, pr)[1]


def test_verdict_instabil_lines():
    base = {"punkte": 624, "clip_n": 332, "mpc_n": 494, "steuerbar_n": 494, "mpc_nicht_steuerbar": 0, "steuerbar_nicht_mpc": 0, "steuerbar_nicht_clip": 162, "clip_nicht_mpc": 0}
    kind, text = ev.verdict_instabil(base, 1.2, 20)
    assert kind == "success" and "Abschneiden konvergiert in 332, MPC in 494" in text and "494 steuerbar" in text and "Abschneiden verfehlt 162" in text
    gleich = dict(base, clip_n=624, mpc_n=624, steuerbar_n=624, steuerbar_nicht_clip=0)
    kind, text = ev.verdict_instabil(gleich, 1.05, 20)
    assert kind == "info" and "keinen Unterschied" in text
    assert ev.verdict_instabil(None, 1.2, 20)[0] == "warning"


def test_verdict_start_cases():
    A, B, Q = D.plant("wagen")

    def lp(cost, end=1e-10):
        return D.Loop(cost, end, 1.0, 5)

    assert ev.verdict_start({"mpc": lp(9.0), "clip": lp(10.0)}, S)[0] == "success"
    assert "**11,11 %**" in ev.verdict_start({"mpc": lp(9.0), "clip": lp(10.0)}, S)[1]
    assert ev.verdict_start({"mpc": lp(10.0), "clip": lp(10.0)}, S)[0] == "info"
    kind, text = ev.verdict_start({"mpc": lp(11.0), "clip": lp(10.0)}, S)
    assert kind == "warning" and "um 9,09 % billiger" in text
    assert ev.verdict_start({"mpc": lp(1e30, end=5.0), "clip": lp(10.0)}, S)[0] == "warning"
    assert ev.verdict_start({"mpc": lp(0.0), "clip": lp(0.0)}, S)[0] == "info"


def test_verdict_qp_names_iterations_and_active_stellsignale():
    import msg_qp as QP

    tr = {"result": QP.QPResult(np.zeros(10), 9), "aktiv": 4, "N": 10, "x0": [10.0, 0.0], "wert": 375.43}
    kind, text = ev.verdict_qp(tr, S)
    assert kind == "info" and "9 Iterationen" in text and "4 von 10 Stellsignalen" in text and "V = 375,43" in text
    assert ev.verdict_qp(dict(tr, result=QP.QPResult(np.zeros(10), -1, True)), S)[0] == "warning"
    # andere Einstellung: keine Zahlen der vorgerechneten Aufwandsmessung
    assert "vorgerechnet" not in ev.verdict_qp(tr, ev.Settings(umax=0.5))[1]


def test_verdict_scaling_names_the_deviation_and_split():
    sc = {"faktoren": [0.25, 2.0], "max_abw": 2.2e-16}
    ab = {"max_nah": 0.83, "max_fern": 3.19, "median_abstand": 5.9}
    kind, text = ev.verdict_scaling(sc, ab, S)
    assert kind == "info" and "2,2e-16" in text and "Maximum 0,83 % in der näheren, 3,19 % in der ferneren Hälfte" in text and "Median des Abstands 5,9" in text
    assert "0,25, 2,00" in text
