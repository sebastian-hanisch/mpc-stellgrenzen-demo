"""Auswertung: Einstellungen, die Rechnungen der sechs Experimente und die Verdikte (reine Funktionen `verdict_*` → (Art, Text)).

Live gerechnet wird, was in unter etwa 2 Sekunden fertig ist (ein QP, ein Regelkreis, das Gitter von 356 Startzuständen); die schweren Messreihen liegen
vorgerechnet in `data/msg_frozen_*.json` (erzeugt von `tools/make_frozen.py`) und sind in der App als „vorgerechnet“ gekennzeichnet.
Alle Zahlen gelten in diesem Modell: x⁺ = A x + B u, Wagen A = [[a, 1], [0, 1]], B = (0; 1), Q = I, Kosten Σ(x'Qx + R u²), |u| ≤ u_max, keine Zustandsgrenzen."""

import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

import msg_constants as C
import msg_dynamics as D
import msg_exact as E
import msg_qp as QP

DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------------------------------------------- Formate
def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def zahl(n: int) -> str:
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(n):,}".replace(",", " ")


def sci(x: float, nd: int = 1) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma (0 bleibt 0)."""
    return "0" if x == 0 else f"{x:.{nd}e}".replace(".", ",")


def frac_text(q: Fraction) -> str:
    """Bruch als Text mit echtem Minuszeichen."""
    t = str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"
    return t.replace("-", "−")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    umax: float = C.DEFAULT_UMAX
    r: float = C.DEFAULT_R
    terminal: str = C.DEFAULT_TERMINAL
    x1: float = C.DEFAULT_X1
    x2: float = C.DEFAULT_X2
    e1_x: float = C.DEFAULT_E1
    e2_system: str = C.DEFAULT_SYSTEM
    e6_a: float = C.DEFAULT_A
    e6_n: int = C.DEFAULT_E6_N


@lru_cache(maxsize=1)
def frozen() -> dict:
    """Alle vorgerechneten Messreihen, ein Eintrag je Datei `msg_frozen_<teil>.json` (fehlt eine Datei, fehlt der Eintrag)."""
    out = {}
    for p in sorted(DATA.glob("msg_frozen_*.json")):
        out[p.stem.removeprefix("msg_frozen_")] = json.loads(p.read_text(encoding="utf-8"))
    return out


def ref_key(r: float, umax: float) -> str:
    return f"{r:g}|{umax:g}"


# ---------------------------------------------------------------------------------------------------------------- Experiment 1: Skalar von Hand
def hand(x) -> dict:
    """Skalar-Handbeispiel (Rawlings Bsp. 2.5, N = 2, mit Faktor ½) exakt: QP-Lösung, vorhergesagter Wert, gefahrene Kosten, Zustandsfolge."""
    xq = Fraction(x)
    u, v, at = E.scalar_n2(xq)
    gefahren, folge = E.scalar_loop(xq)
    return {"x": xq, "u": u, "V": v, "at": at, "gefahren": gefahren, "folge": folge, "u0_gesetz": -E.sat(Fraction(3, 5) * xq)}


def hand_cost_path(h) -> list[float]:
    """Summe der Stufenkosten (Faktor ½) entlang des Regelkreises, k = 0 … 40, als Gleitkomma für das Diagramm."""
    x = float(h["x"])
    cum, out = 0.0, [0.0]
    for _ in range(40):
        u = -max(-1.0, min(1.0, 0.6 * x))
        cum += 0.5 * (x * x + u * u)
        x += u
        out.append(cum)
    return out


def verdict_hand(h) -> tuple[str, str]:
    x, u, v, g = h["x"], h["u"], h["V"], h["gefahren"]
    if x == 0:
        return "info", "Bei x = 0 ist u = (0, 0), die Kosten sind 0: es gibt nichts zu regeln."
    aktiv = [i for i, a in enumerate(h["at"]) if a != "frei"]
    if not aktiv:
        grenze = "keine Grenze greift (|x| ≤ 5/3)"
    elif aktiv == [0, 1]:
        grenze = "beide Stellsignale liegen an der Grenze (|x| ≥ 3)"
    else:
        grenze = "nur u₀ liegt an der Grenze (5/3 ≤ |x| ≤ 3)"
    pct = 100 * float(g - v) / float(g)
    return "info", (f"x = {frac_text(x)}: u = ({frac_text(u[0])}; {frac_text(u[1])}), {grenze}; u₀ = −sat(3x/5) = {frac_text(h['u0_gesetz'])}. "
                    f"Vorhergesagt (zwei Schritte plus Endkosten, Faktor ½) V = {frac_text(v)} = {de(float(v), 2)}, gefahren bis ins Unendliche {frac_text(g)} = {de(float(g), 2)}: "
                    f"die Voraussage liegt **{de(pct, 0)} %** unter den gefahrenen Kosten, weil nach zwei Schritten nicht Schluss ist.")


# ---------------------------------------------------------------------------------------------------------------- Experiment 2: Abschneiden gegen MPC
def problem(s: Settings, kind="wagen", a=1.0, N=None, umax=None, terminal=None, r=None) -> D.MPCProblem:
    return D.build_mpc(kind, float(a), int(s.n if N is None else N), float(s.r if r is None else r), float(s.umax if umax is None else umax),
                       s.terminal if terminal is None else terminal)


def wagen_points():
    return D.grid_points(C.GRID_X1, C.GRID_X2)


def wagen_grid(s: Settings) -> D.GridRun:
    """Gefahrene Kosten von Abschneiden und MPC auf den 356 Startzuständen (live, etwa 1 bis 3 Sekunden bei N = 10)."""
    return D.run_grid(problem(s), wagen_points())


def wagen_counts(g: D.GridRun) -> dict:
    ok = (g.end_mpc < C.CONV_COUNT_TOL) & (g.end_clip < C.CONV_COUNT_TOL)
    ch, eq, de_, adv, dis = D.compare_counts(g.cost_clip[ok], g.cost_mpc[ok])
    return {"punkte": len(g.pts), "verglichen": int(ok.sum()), "mpc_konvergiert": int(np.sum(g.end_mpc < C.CONV_COUNT_TOL)),
            "clip_konvergiert": int(np.sum(g.end_clip < C.CONV_COUNT_TOL)), "billiger": ch, "gleich": eq, "teurer": de_, "vorteil_max": adv, "nachteil_max": dis}


def wagen_mehrkosten(g: D.GridRun) -> np.ndarray:
    """Relative Mehrkosten von Abschneiden gegen MPC in % (positiv: Abschneiden teurer), je Startzustand."""
    return 100.0 * (g.cost_clip / g.cost_mpc - 1.0)


def firstu_check(s: Settings) -> dict:
    """Erstes Stellsignal des QP gegen das abgeschnittene erste Stellsignal der unbeschränkten N-Schritt-Lösung (Gitter 41 × 33 = 1 353 Punkte)."""
    prob = problem(s)
    k = D.unconstrained_first_gain(prob)
    pts = D.grid_points(C.FIRSTU_X1, C.FIRSTU_X2, drop_origin=False)
    law = D.MPCLaw(prob)
    dev = np.array([abs(law(np.array(p)) - D.clip_value(float(k @ np.array(p)), prob.umax)) for p in pts])
    return {"punkte": len(pts), "abweichend": int(np.sum(dev > 1e-6)), "groesste": float(dev.max())}


def skalar_firstu(s: Settings) -> dict:
    """Skalar: erstes Stellsignal von MPC gegen das abgeschnittene unbeschränkte Gesetz auf 2 401 Punkten, dazu gefahrene Kosten ab x1 (Faktor ½)."""
    prob = problem(s, kind="skalar")
    ku = float(D.unconstrained_first_gain(prob)[0])
    xs = D.grid_axis(C.SCALAR_GRID)
    law_u = np.array([QP.qp_active_set(prob.H, prob.F[:, 0] * x, prob.lo, prob.hi).u[0] for x in xs])
    clip_u = np.clip(ku * xs, -prob.umax, prob.umax)
    dev = np.abs(law_u - clip_u)
    x0 = np.array([float(s.x1)])
    c_mpc = 0.5 * D.run_loop(D.MPCLaw(prob), x0, prob.A, prob.B, prob.Q, prob.R, tol=1e-10).cost
    c_clip = 0.5 * D.run_loop(D.ClipLaw(prob.K, prob.umax), x0, prob.A, prob.B, prob.Q, prob.R, tol=1e-10).cost
    return {"xs": xs, "mpc": law_u, "clip": clip_u, "punkte": len(xs), "abweichend": int(np.sum(dev > 1e-9)), "groesste": float(dev.max()), "gewinn": -ku,
            "gewinn_dare": float(prob.K[0, 0]), "c_mpc": c_mpc, "c_clip": c_clip, "x0": float(s.x1)}


def verdict_skalar(r: dict, s: Settings) -> tuple[str, str]:
    basis = f"Skalar, N = {s.n}, Endkosten {C.TERMINAL_LABELS[s.terminal]}, Grenze {de(s.umax, 2)}, R = {de(s.r, 1)}"
    if r["abweichend"] == 0:
        satz = (f"{basis}: das erste Stellsignal des QP stimmt mit dem abgeschnittenen unbeschränkten Gesetz u = clip({de(-r['gewinn'], 4)}·x) "
                f"in **{zahl(r['punkte'])} von {zahl(r['punkte'])}** Punkten überein (größte Abweichung {sci(r['groesste'])}). Hier bringt das QP gegenüber dem Abschneiden nichts. ")
    else:
        satz = f"{basis}: das erste Stellsignal weicht in {zahl(r['abweichend'])} von {zahl(r['punkte'])} Punkten ab (größte Abweichung {de(r['groesste'], 3)}). "
    satz += (f"Gefahrene Kosten ab x = {de(r['x0'], 1)} (Faktor ½): MPC {de(r['c_mpc'], 4)}, LQR-Gesetz abgeschnitten (Gewinn {de(r['gewinn_dare'], 4)}) {de(r['c_clip'], 4)}. "
             "Warum es im Skalar keinen Unterschied gibt, ist nicht erklärt, nur gemessen.")
    return "info", satz


def verdict_wagen(c: dict, s: Settings, firstu: dict | None = None) -> tuple[str, str]:
    basis = f"Wagen, u_max = {de(s.umax, 2)}, R = {de(s.r, 1)}, N = {s.n}, Endkosten {C.TERMINAL_LABELS[s.terminal]}"
    zahlen = f"{zahl(c['verglichen'])} Startzustände"
    if c["mpc_konvergiert"] < c["punkte"] or c["clip_konvergiert"] < c["punkte"]:
        zusatz = f" MPC konvergiert in {zahl(c['mpc_konvergiert'])} von {zahl(c['punkte'])} Startzuständen, Abschneiden in {zahl(c['clip_konvergiert'])}; verglichen werden nur Zustände, in denen beide konvergieren."
    else:
        zusatz = ""
    fu = ""
    if firstu is not None:
        fu = f" Das erste Stellsignal weicht von der abgeschnittenen unbeschränkten Lösung in {zahl(firstu['abweichend'])} von {zahl(firstu['punkte'])} Gitterpunkten ab (größte Abweichung {de(firstu['groesste'], 3)})."
    if c["verglichen"] == 0:
        return "warning", f"{basis}: kein Startzustand, in dem beide Regeln konvergieren.{zusatz}"
    if c["teurer"] > 0:
        return "warning", (f"{basis}: MPC ist in {zahl(c['teurer'])} von {zahl(c['verglichen'])} Startzuständen **teurer** als LQR-dann-abschneiden (größter Nachteil {de(c['nachteil_max'], 2)} %), "
                           f"billiger in {zahl(c['billiger'])}, gleich in {zahl(c['gleich'])}.{zusatz}{fu}")
    if c["billiger"] == 0:
        return "info", f"{basis}: kein Unterschied. Gleiche Kosten in {zahl(c['gleich'])} von {zahl(c['verglichen'])} Startzuständen, MPC nirgends billiger.{zusatz}{fu}"
    return "success", (f"{basis}: MPC ist in {zahl(c['billiger'])} von {zahl(c['verglichen'])} Startzuständen billiger als LQR-dann-abschneiden, gleich in {zahl(c['gleich'])}, "
                       f"teurer in {zahl(c['teurer'])}; größter Vorteil {de(c['vorteil_max'], 2)} % ({zahlen}, Gitter x₁ −10 bis 10, x₂ −4 bis 4).{zusatz}{fu}")


# ---------------------------------------------------------------------------------------------------------------- Experiment 3: nur das Verhältnis zählt
def scaling_check(s: Settings) -> dict:
    """Kosten von (c·x0, c·u_max) geteilt durch c² und durch die Kosten von (x0, u_max), für Abschneiden und MPC; Startzustand aus der Seitenleiste."""
    x0 = np.array([s.x1, s.x2])
    if not np.any(x0):
        x0 = np.array([4.0, 1.0])
    out = {"faktoren": list(C.SCALING_FACTORS), "clip": [], "mpc": [], "x0": [float(v) for v in x0]}
    for name in ("clip", "mpc"):
        base = problem(s)
        law = D.ClipLaw(base.K, s.umax) if name == "clip" else D.MPCLaw(base)
        c0 = D.run_loop(law, x0, base.A, base.B, base.Q, base.R).cost
        for c in C.SCALING_FACTORS:
            pc = problem(s, umax=c * s.umax)
            lawc = D.ClipLaw(pc.K, c * s.umax) if name == "clip" else D.MPCLaw(pc)
            cc = D.run_loop(lawc, c * x0, pc.A, pc.B, pc.Q, pc.R).cost
            out[name].append(cc / (c * c * c0))
    out["max_abw"] = max(abs(v - 1.0) for name in ("clip", "mpc") for v in out[name])
    return out


def abstand_aufschlag(g: D.GridRun, umax: float) -> dict:
    """Mehrkosten von Abschneiden gegen MPC (%) über dem Abstand ‖x0‖/u_max; Maximum in der näheren und in der ferneren Hälfte der Abstände."""
    dist = np.linalg.norm(g.pts, axis=1) / umax
    mk = wagen_mehrkosten(g)
    ok = (g.end_mpc < C.CONV_COUNT_TOL) & (g.end_clip < C.CONV_COUNT_TOL)
    d, m = dist[ok], mk[ok]
    med = float(np.median(d)) if len(d) else 0.0
    nah, fern = m[d <= med], m[d > med]
    return {"dist": d, "mehrkosten": m, "median_abstand": med, "max_nah": float(nah.max()) if len(nah) else 0.0, "max_fern": float(fern.max()) if len(fern) else 0.0}


def verdict_scaling(sc: dict, ab: dict, s: Settings) -> tuple[str, str]:
    fz = frozen().get("d1", {}).get("szenarien", {})
    e1, e2 = fz.get(ref_key(1.0, 1.0)), fz.get(ref_key(1.0, 0.5))
    teil = ""
    if e1 and e2:
        teil = (f" Aufschlag von „LQR, dann abschneiden“ gegen die Referenz (vorgerechnet, R = 1, 356 Startzustände): Maximum {de(e1['clip']['max'], 2)} % bei u_max = 1, "
                f"{de(e2['clip']['max'], 2)} % bei u_max = 0,5.")
    return "info", (f"Kosten von (c·x₀, c·u_max) geteilt durch c² und durch die Kosten von (x₀, u_max), Faktoren {', '.join(de(c, 2) for c in sc['faktoren'])}: "
                    f"größte Abweichung von 1 gleich {sci(sc['max_abw'])} (Abschneiden und MPC, N = {s.n}). Nur das Verhältnis Zustand zu Grenze zählt. "
                    f"Mehrkosten des Abschneidens gegen MPC im Gitter: Maximum {de(ab['max_nah'], 2)} % in der näheren, {de(ab['max_fern'], 2)} % in der ferneren Hälfte der Abstände ‖x₀‖/u_max "
                    f"(Median des Abstands {de(ab['median_abstand'], 1)}).{teil}")


# ---------------------------------------------------------------------------------------------------------------- Experiment 4: Horizont und Endkosten
def horizon_stats(s: Settings, g: D.GridRun) -> dict | None:
    """Aufschlag von MPC (N, Endkosten) gegen die Referenz (N = 60, DARE; vorgerechnet) über das Gitter; None, wenn die Referenz für (R, u_max) fehlt."""
    ref = frozen().get("d1", {}).get("ref_costs", {}).get(ref_key(s.r, s.umax))
    if ref is None or len(ref) != len(g.pts):
        return None
    ref = np.array(ref)
    ok = g.end_mpc < C.CONV_COUNT_TOL
    out = {"punkte": len(ref), "konvergiert": int(ok.sum())}
    if ok.any():
        a = 100 * (g.cost_mpc[ok] / ref[ok] - 1)
        out.update({"median": float(np.median(a)), "max": float(np.max(a))})
    return out


def prediction(s: Settings) -> dict:
    """Vorhergesagter Wert V_N(x0) des QP gegen die gefahrenen Kosten (N aus der Seitenleiste und die Referenz N = 60), Startzustand aus der Seitenleiste."""
    x0 = np.array([s.x1, s.x2])
    out = {"x0": [s.x1, s.x2]}
    for tag, N, term in (("n", s.n, s.terminal), ("ref", C.REF_N, "DARE")):
        prob = problem(s, N=N, terminal=term)
        r = QP.qp_solve(prob.H, prob.F @ x0, prob.lo, prob.hi)
        out[tag] = {"N": N, "V": QP.qp_value(r.u, prob.H, prob.F @ x0, float(x0 @ prob.Cmat @ x0)),
                    "gefahren": D.run_loop(D.MPCLaw(prob), x0, prob.A, prob.B, prob.Q, prob.R).cost}
    return out


def verdict_horizon(s: Settings, st: dict | None, pr: dict) -> tuple[str, str]:
    basis = f"u_max = {de(s.umax, 2)}, R = {de(s.r, 1)}, N = {s.n}, Endkosten {C.TERMINAL_LABELS[s.terminal]}"
    if st is None:
        teil = "Die Referenz (N = 60) für diese Kombination ist nicht vorgerechnet."
    elif st["konvergiert"] == 0:
        teil = f"MPC konvergiert in keinem der {zahl(st['punkte'])} Startzustände."
    else:
        teil = (f"Aufschlag gegen die Referenz über {zahl(st['konvergiert'])} konvergierende von {zahl(st['punkte'])} Startzuständen: Median {de(st['median'], 2)} %, Maximum {de(st['max'], 2)} %.")
        if st["konvergiert"] < st["punkte"]:
            teil += f" In {zahl(st['punkte'] - st['konvergiert'])} Startzuständen konvergiert MPC nicht."
    x0 = pr["x0"]
    n, ref = pr["n"], pr["ref"]
    vergleich = f"Start ({de(x0[0], 1)}; {de(x0[1], 1)}): vorhergesagt V = {de(n['V'], 2)}, gefahren {de(n['gefahren'], 2)}; Referenz N = 60: vorhergesagt {de(ref['V'], 2)}, gefahren {de(ref['gefahren'], 2)}."
    art = "warning" if (st is not None and (st["konvergiert"] < st["punkte"] or st.get("max", 0) > 1.0)) else "info"
    return art, f"{basis}: {teil} {vergleich}"


# ---------------------------------------------------------------------------------------------------------------- Experiment 5: das QP anschauen
def qp_trace(s: Settings) -> dict:
    prob = problem(s)
    x0 = np.array([s.x1, s.x2])
    law = D.MPCLaw(prob)
    r = law.solve(x0, trace=True)
    unc = -np.linalg.solve(prob.H, prob.F @ x0)
    const = float(x0 @ prob.Cmat @ x0)
    return {"result": r, "unbeschraenkt": unc, "umax": prob.umax, "N": prob.N, "const": const, "x0": [s.x1, s.x2],
            "aktiv": int(np.sum(np.isclose(np.abs(r.u), prob.umax, rtol=0, atol=1e-9 * prob.umax))), "wert": QP.qp_value(r.u, prob.H, prob.F @ x0, const)}


def trace_table(tr: dict) -> list[dict]:
    rows = []
    for st in tr["result"].trace:
        rows.append({"Iteration": st["iteration"], "unten fest": ", ".join(f"u{i}" for i in np.nonzero(st["unten"])[0]) or "–",
                     "oben fest": ", ".join(f"u{i}" for i in np.nonzero(st["oben"])[0]) or "–", "Zielwert V": st["wert"] + tr["const"], "Aktion": st.get("aktion", "")})
    return rows


def verdict_qp(tr: dict, s: Settings) -> tuple[str, str]:
    r = tr["result"]
    if r.fallback:
        return "warning", "Die Aktivmengen-Methode ist nicht zu Ende gekommen; die Lösung stammt aus dem Rückfall (bvls)."
    d7 = frozen().get("d1", {}).get("d7", {}).get(str(s.n))
    teil = ""
    if d7 and s.umax == 1.0 and s.r == 1.0 and s.terminal == "DARE":
        teil = (f" Über die {zahl(d7['aufrufe'])} QP-Aufrufe entlang der 356 Bahnen (u_max = 1, R = 1, Kaltstart, vorgerechnet): im Mittel {de(d7['iter_mittel'], 2)} Iterationen, "
                f"höchstens {d7['iter_max']}; in {de(100 * d7['anteil_aktiv'], 1)} % der Aufrufe ist mindestens eine Grenze aktiv; {d7['muster']} verschiedene Vorzeichenmuster in den Bahnen.")
    return "info", (f"Start ({de(tr['x0'][0], 1)}; {de(tr['x0'][1], 1)}), N = {tr['N']}, Kaltstart: {r.iterations} Iterationen, {tr['aktiv']} von {tr['N']} Stellsignalen an der Grenze, "
                    f"vorhergesagter Wert V = {de(tr['wert'], 2)}.{teil}")


def wagen_hand() -> dict:
    return E.wagen_hand()


# ---------------------------------------------------------------------------------------------------------------- Experiment 6: instabile Strecke
def instabil_maps(a: float, N: int) -> dict | None:
    fz = frozen().get("instabil")
    if fz is None:
        return None
    e = fz["a"].get(f"{a:g}")
    if e is None or str(N) not in e["mpc"]:
        return None
    n = len(D.grid_axis(C.UNSTABLE_GRID))
    pts = D.grid_points(C.UNSTABLE_GRID, C.UNSTABLE_GRID)
    clip, mpc, ctrl = np.array(e["clip"], bool), np.array(e["mpc"][str(N)], bool), np.array(e["steuerbar"], bool)
    return {"pts": np.array(pts), "n": n, "clip": clip, "mpc": mpc, "steuerbar": ctrl, "punkte": len(pts),
            "clip_n": int(clip.sum()), "mpc_n": int(mpc.sum()), "steuerbar_n": int(ctrl.sum()), "mpc_nicht_steuerbar": int(np.sum(mpc & ~ctrl)),
            "steuerbar_nicht_mpc": int(np.sum(ctrl & ~mpc)), "steuerbar_nicht_clip": int(np.sum(ctrl & ~clip)), "clip_nicht_mpc": int(np.sum(clip & ~mpc))}


def verdict_instabil(m: dict | None, a: float, N: int) -> tuple[str, str]:
    if m is None:
        return "warning", "Für diese Kombination liegt keine vorgerechnete Messung vor."
    basis = f"a = {de(a, 2)}, N = {N}, Grenze 1, R = 1, Endkosten DARE, Gitter −6 bis 6 (Schritt 0,5), {m['punkte']} Startzustände"
    text = (f"{basis}: Abschneiden konvergiert in {m['clip_n']}, MPC in {m['mpc_n']}; per linearem Programm (100 Schritte, |u| ≤ 1) sind {m['steuerbar_n']} steuerbar. "
            f"MPC konvergiert in {m['mpc_nicht_steuerbar']} nicht steuerbaren Punkten und verfehlt {m['steuerbar_nicht_mpc']} steuerbare; "
            f"Abschneiden verfehlt {m['steuerbar_nicht_clip']} steuerbare Punkte; Abschneiden konvergiert in {m['clip_nicht_mpc']} Punkten, in denen MPC nicht konvergiert.")
    if m["steuerbar_nicht_clip"] == 0 and m["steuerbar_nicht_mpc"] == 0:
        return "info", text + " Auf diesem Gitter gibt es bei dieser Strecke keinen Unterschied."
    return ("success" if m["steuerbar_nicht_mpc"] < m["steuerbar_nicht_clip"] else "info"), text


# ---------------------------------------------------------------------------------------------------------------- Primäransicht: ein Startzustand
def trajectory(s: Settings) -> dict:
    """Regelkreis aus dem Startzustand der Seitenleiste für beide Regeln (Wagen, Einstellungen der Seitenleiste)."""
    prob = problem(s)
    x0 = np.array([s.x1, s.x2])
    law = D.MPCLaw(prob)
    mpc = D.run_loop(law, x0, prob.A, prob.B, prob.Q, prob.R, record=True)
    clip = D.run_loop(D.ClipLaw(prob.K, prob.umax), x0, prob.A, prob.B, prob.Q, prob.R, record=True)
    return {"mpc": mpc, "clip": clip, "umax": prob.umax, "x0": x0}


def verdict_start(tr: dict, s: Settings) -> tuple[str, str]:
    """Verdikt der Primäransicht: gefahrene Kosten beider Regeln aus dem Startzustand."""
    m, c = tr["mpc"], tr["clip"]
    start = f"Start ({de(s.x1, 1)}; {de(s.x2, 1)})"
    if m.end_norm >= C.CONV_COUNT_TOL or c.end_norm >= C.CONV_COUNT_TOL:
        wer = "MPC" if m.end_norm >= C.CONV_COUNT_TOL else "Abschneiden"
        return "warning", f"{start}: {wer} erreicht den Ursprung in {C.STEPS_MAX} Takten nicht (Endnorm {sci(m.end_norm)} bzw. {sci(c.end_norm)})."
    if c.cost <= 0:
        return "info", f"{start}: der Zustand liegt schon im Ursprung, die Kosten sind 0."
    rel = 100 * (c.cost / m.cost - 1)
    if abs(rel) <= 100 * C.EQUAL_REL:
        return "info", f"{start}: beide Regeln kosten dasselbe ({de(m.cost, 2)}); hier macht das QP keinen Unterschied."
    if rel > 0:
        return "success", f"{start}: MPC kostet {de(m.cost, 2)}, LQR-dann-abschneiden {de(c.cost, 2)}, also **{de(rel, 2)} %** mehr als MPC."
    return "warning", f"{start}: MPC kostet {de(m.cost, 2)}, LQR-dann-abschneiden {de(c.cost, 2)}; Abschneiden ist hier um {de(-rel, 2)} % billiger."


def limits_rows() -> list[tuple[str, str, str]]:
    """Zeilen der Tabelle „Wo die Annahmen enden“; Zahlen aus den vorgerechneten Messreihen, damit Text und Messung nicht auseinanderlaufen."""
    fz = frozen()
    rf = fz.get("rueckfall", {}).get("1.2|40")
    weit = fz.get("weit", {})
    zeile_rueck = "Bei N = 40 und a = 1,2 (Konditionszahl von H sehr groß) bricht die Aktivmengen-Methode ab"
    if rf and rf.get("aufrufe"):
        zeile_rueck += f" ({rf['rueckfaelle']} von {zahl(rf['aufrufe'])} Aufrufen, vorgerechnet, Rückfall auf bvls)"
    zeile_weit = "Abschneiden konvergiert auch aus sehr großen Zuständen (kein Stabilitätsunterschied am Doppelintegrator), überschwingt aber stark"
    if weit.get("zufall"):
        z = weit["zufall"]
        zeile_weit += (f": im Median das {de(z['verhaeltnis_median'], 1)}-Fache, höchstens das {de(z['verhaeltnis_max'], 1)}-Fache der Startnorm ({z['starts']} Zufallsstarts, u_max = 1, R = 1)")
    return [
        ("Nur Boxgrenzen auf u; keine Zustandsgrenzen, keine Terminalmenge", "Mit Zustandsgrenzen kann das QP unzulässig werden; hier ist es immer lösbar.", "[`mpc-zulaessigkeit-demo`](https://github.com/sebastian-hanisch/mpc-zulaessigkeit-demo)"),
        ("Ein Modell (Wagen), dazu die Familie a = 1 bis 1,25 und der Skalar", "Andere Strecken, mehrere Eingänge, Reibung und andere Gewichte Q sind nicht gemessen.", "–"),
        ("Referenz = MPC mit N = 60 (Endkosten DARE)", "Kein bewiesener unendlicher Horizont; gestützt durch den Vergleich von vorhergesagtem und gefahrenem Wert und den Skalar mit dynamischer Programmierung.", "–"),
        ("Große Zustände", zeile_weit + ".", "–"),
        ("Kein Rauschen, keine Störung, kein Verzug, kein Modellfehler", "Mit Stellgrenze und Störung oder Verzug nicht gemessen.",
         "[`plan-stoerung-demo`](https://github.com/sebastian-hanisch/plan-stoerung-demo), [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo), [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo)"),
        ("Horizont höchstens 20 in der App", zeile_rueck + ".", "–"),
        ("Rechenzeit", "Nicht gemessen und nicht festgeschrieben; Warmstart und Vergleich mit Innere-Punkte-Verfahren oder ADMM kommen nicht vor.", "Linie Konvexe und konische Optimierung"),
    ]


def gegenprobe_text() -> str:
    """Satz zur Gegenprobe „MPC nie teurer“ aus der vorgerechneten Messung (1 000 Zufallsinstanzen, Zufalls-Seed 14); leer, wenn die Datei fehlt."""
    g = frozen().get("gegenprobe")
    if not g:
        return ""
    teurer = sum(v["teurer"] for v in g["n"].values())
    return (f"Gegenprobe (vorgerechnet): {zahl(g['instanzen'])} Zufallsinstanzen (R, u_max, x₀ zufällig, Zufalls-Seed {g['seed']}), MPC mit N = 2, 3 und 5 und Endkosten DARE: "
            f"{teurer} Fälle, in denen MPC teurer ist als Abschneiden. Eine Garantie ist das nicht.")
