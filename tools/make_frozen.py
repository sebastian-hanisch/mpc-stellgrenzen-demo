"""Erzeugt `data/msg_frozen_<teil>.json`: die Messreihen, die die App nicht live rechnet (Laufzeit je Teil Sekunden bis einige Minuten).

Aufruf (Wurzelordner der Demo): _venvs/test/Scripts/python.exe -X utf8 tools/make_frozen.py <teil> [<teil> ...]   oder: alle
Teile:
  kurz         Skalierung (D2), Voraussage gegen gefahren (D3), erster Gewinn ohne Grenze (D4), Erststellsignal QP gegen Abschneiden (D5)
  d1           gefahrene Kosten am Wagen, 356 Startzustände: drei Szenarien, Horizont N = 1 bis 20, Endkosten DARE; ohne Endkosten (D6);
               Aufwand der Aktivmengen-Methode (D7); Schwellen-Zählung; Referenzkosten (N = 60) für alle Regler-Kombinationen der App
  skalar       Skalar: Erststellsignal MPC gegen abgeschnittene unbeschränkte Folge (2 401 Punkte, N = 1 bis 8, zwei Endkosten), gefahrene Kosten, DP-Gitter
  gegenprobe   1 000 Zufallsinstanzen (Zufalls-Seed 14): ist MPC teurer als Abschneiden?
  weit         Abschneiden und MPC bei sehr großen Zuständen (Überschwingen)
  instabil     offen instabile Strecke: Konvergenz-Karten (Abschneiden, MPC N = 3/5/10/20) und die per LP steuerbare Menge, je a
  orakel       Löser gegen Aufzählung aller Muster, bvls und SLSQP
  rueckfall    Wie oft bricht die Aktivmengen-Methode ab (Rückfall)?
Die Zufallszahlen kommen aus dem SplitMix64 der Demo (msg_rng.py), nie aus Python-`hash()`, `random` oder `math.log`/`math.cos`.
Messgrößen und Zähleinheiten stehen in der README unter „Befunde“; die Zahlen dieser Dateien sind die dort genannten Messungen."""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import msg_constants as C  # noqa: E402
import msg_dynamics as D  # noqa: E402
import msg_qp as QP  # noqa: E402
from msg_rng import SplitMix64  # noqa: E402

DATA = ROOT / "data"
SEED = 14
NS = (1, 2, 3, 4, 5, 7, 10, 20)


def save(teil, obj):
    DATA.mkdir(exist_ok=True)
    (DATA / f"msg_frozen_{teil}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"gespeichert: data/msg_frozen_{teil}.json")


def pts_wagen():
    return D.grid_points(C.GRID_X1, C.GRID_X2)


def stats_vs_ref(cost, end, ref):
    ok = end < C.CONV_COUNT_TOL
    auf = 100 * (cost / ref - 1)
    if not ok.any():
        return {"konvergiert": 0}
    a = auf[ok]
    return {"konvergiert": int(ok.sum()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(np.max(a))}


# ---------------------------------------------------------------------------------------------------------------- kurz
def teil_kurz():
    out = {}
    # D2 Skalierung
    P, K = D.dare(*D.plant("wagen"), 1.0)
    worst = 0.0
    for x0 in ([4.0, 1.0], [-7.0, 2.5], [9.0, -3.0], [2.0, 2.0]):
        for c in (0.25, 0.5, 3.0, 8.0):
            for name in ("clip", "mpc"):
                costs = []
                for um, sc in ((1.0, 1.0), (c, c)):
                    prob = D.build_mpc("wagen", 1.0, 10, 1.0, um, "DARE")
                    law = D.ClipLaw(prob.K, um) if name == "clip" else D.MPCLaw(prob)
                    costs.append(D.run_loop(law, sc * np.array(x0), prob.A, prob.B, prob.Q, 1.0).cost)
                worst = max(worst, abs(costs[1] / (c * c * costs[0]) - 1))
    out["d2_skalierung_max_abw"] = worst
    # D3 Voraussage gegen gefahren
    d3 = {}
    for x0 in ([10.0, 0.0], [-10.0, 4.0], [5.0, -2.0], [2.0, 1.0]):
        row = {}
        for N in (2, 5, 10, 60):
            prob = D.build_mpc("wagen", 1.0, N, 1.0, 1.0, "DARE")
            x = np.array(x0)
            r = QP.qp_solve(prob.H, prob.F @ x, prob.lo, prob.hi)
            v = QP.qp_value(r.u, prob.H, prob.F @ x, float(x @ prob.Cmat @ x))
            gef = D.run_loop(D.MPCLaw(prob), x, prob.A, prob.B, prob.Q, 1.0).cost
            row[str(N)] = {"vorhergesagt": v, "gefahren": gef}
        d3[f"{x0[0]:g};{x0[1]:g}"] = row
    out["d3"] = d3
    # D4 erster unbeschränkter Gewinn
    d4 = {}
    for R in (1.0, 0.1):
        _, Kd = D.dare(*D.plant("wagen"), R)
        zd, zq = [], []
        for N in range(1, 11):
            for term, ziel in (("DARE", zd), ("Q", zq)):
                prob = D.build_mpc("wagen", 1.0, N, R, 1.0, term)
                ziel.append(float(np.max(np.abs(D.unconstrained_first_gain(prob) + Kd[0]))))
        d4[str(R)] = {"dare_max": max(zd), "q": {str(n): zq[n - 1] for n in (1, 2, 5, 10)}}
    out["d4"] = d4
    prob = D.build_mpc("wagen", 1.0, 10, 1.0, 1.0, "DARE")
    law = D.MPCLaw(prob)
    xs = [np.array([a, b]) for a in np.arange(-3.0, 3.0 + 1e-9, 0.25) for b in np.arange(-1.5, 1.5 + 1e-9, 0.25)]
    out["d4_innen"] = {"gleich": int(sum(abs(law(x) + (prob.K @ x)[0]) < 1e-9 for x in xs)), "von": len(xs)}
    # D5 Erststellsignal QP gegen clip(-Kx)
    pts = D.grid_points(C.FIRSTU_X1, C.FIRSTU_X2, drop_origin=False)
    dev = np.array([abs(law(np.array(p)) - D.clip_value(-(prob.K @ np.array(p))[0], 1.0)) for p in pts])
    out["d5"] = {"punkte": len(pts), "abweichend": int(np.sum(dev > 1e-6)), "groesste": float(dev.max())}
    save("kurz", out)


# ---------------------------------------------------------------------------------------------------------------- d1
def teil_d1():
    t0 = time.time()
    pts = pts_wagen()
    out = {"punkte": len(pts), "szenarien": {}, "d6": {}, "d7": {}, "ref_costs": {}, "schwelle": {}}
    refs = {}
    for R in C.R_CHOICES:
        for um in C.UMAX_CHOICES:
            refs[(R, um)] = D.reference_costs(R, um, pts)
            out["ref_costs"][f"{R:g}|{um:g}"] = [float(v) for v in refs[(R, um)]]
            print("Referenz", R, um, f"{time.time() - t0:.0f} s", flush=True)
    for R, um in ((1.0, 1.0), (1.0, 0.5), (0.1, 1.0)):
        ref = refs[(R, um)]
        prob10 = D.build_mpc("wagen", 1.0, 10, R, um, "DARE")
        cc, ec, _, _ = D.run_clip_batch(np.array(pts), prob10.K, um, prob10.A, prob10.B, prob10.Q, R)
        sc = {"clip": stats_vs_ref(cc, ec, ref), "mpc": {}}
        for N in NS:
            prob = D.build_mpc("wagen", 1.0, N, R, um, "DARE")
            g = D.run_grid(prob, pts)
            ch, eq, de, adv, dis = D.compare_counts(g.cost_clip, g.cost_mpc)
            e = stats_vs_ref(g.cost_mpc, g.end_mpc, ref)
            e.update({"billiger": ch, "gleich": eq, "teurer": de, "vorteil_max": adv, "nachteil_max": dis, "iter_mittel": g.iterations_mean})
            sc["mpc"][str(N)] = e
            if N == 10:
                rel = g.cost_clip / g.cost_mpc - 1       # positiv: Abschneiden teurer
                sc_s = {}
                for s in (1e-12, 1e-9, 1e-6, 1e-4):
                    sc_s[f"{s:g}"] = {"gleich": int(np.sum(np.abs(rel) <= s)), "clip_teurer": int(np.sum(rel > s)), "clip_billiger": int(np.sum(rel < -s))}
                zw = rel[(np.abs(rel) > 1e-12) & (np.abs(rel) < 1e-6)]
                pos = rel[rel > 1e-12]
                out["schwelle"][f"{R:g}|{um:g}"] = {"stufen": sc_s, "dazwischen": int(len(zw)), "kleinster_ueber_1e-12": float(pos.min()) if len(pos) else None}
            print("d1", R, um, N, f"{time.time() - t0:.0f} s", flush=True)
        out["szenarien"][f"{R:g}|{um:g}"] = sc
    # D6 ohne Endkosten
    ref = refs[(1.0, 1.0)]
    for N in (1, 2, 3, 5):
        prob = D.build_mpc("wagen", 1.0, N, 1.0, 1.0, "Q")
        g = D.run_grid(prob, pts)
        ch, eq, de, adv, dis = D.compare_counts(g.cost_clip[g.end_mpc < C.CONV_COUNT_TOL], g.cost_mpc[g.end_mpc < C.CONV_COUNT_TOL])
        e = stats_vs_ref(g.cost_mpc, g.end_mpc, ref)
        e.update({"billiger": ch, "gleich": eq, "teurer": de, "vorteil_max": adv, "nachteil_max": dis})
        out["d6"][str(N)] = e
    # D7 Aufwand
    for N in (5, 10, 20):
        prob = D.build_mpc("wagen", 1.0, N, 1.0, 1.0, "DARE")
        law = D.MPCLaw(prob, record=True)
        for x0 in pts:
            D.run_loop(law, x0, prob.A, prob.B, prob.Q, 1.0)
        its = np.array(law.iterations)
        act = np.array(law.active)
        out["d7"][str(N)] = {"aufrufe": law.calls, "iter_mittel": float(its.mean()), "iter_max": int(its.max()), "anteil_aktiv": float(np.mean(act > 0)), "muster": len(law.patterns),
                             "rueckfaelle": law.fallbacks}
    save("d1", out)


# ---------------------------------------------------------------------------------------------------------------- skalar
def teil_skalar():

    out = {"s3": {}, "s3_versetzt": {}, "s4": {}}
    xs = D.grid_axis(C.SCALAR_GRID)
    for term in ("Q", "DARE"):
        for N in range(1, 9):
            prob = D.build_mpc("skalar", 1.0, N, 1.0, 1.0, term)
            ku = D.unconstrained_first_gain(prob)[0]
            dev = np.array([abs(QP.qp_active_set(prob.H, prob.F[:, 0] * x, prob.lo, prob.hi).u[0] - D.clip_value(ku * x, 1.0)) for x in xs])
            out["s3"][f"{term}|{N}"] = {"gewinn": float(-ku), "punkte": len(xs), "abweichend": int(np.sum(dev > 1e-9)), "groesste": float(dev.max())}
    xs2 = np.round(np.arange(-12.0, 12.0 + 1e-9, 0.013), 10)
    for term in ("Q", "DARE"):
        for N in (1, 2, 5, 8):
            prob = D.build_mpc("skalar", 1.0, N, 1.0, 0.7, term)
            ku = D.unconstrained_first_gain(prob)[0]
            dev = np.array([abs(QP.qp_active_set(prob.H, prob.F[:, 0] * x, prob.lo, prob.hi).u[0] - D.clip_value(ku * x, 0.7)) for x in xs2])
            out["s3_versetzt"][f"{term}|{N}"] = {"punkte": len(xs2), "abweichend": int(np.sum(dev > 1e-9)), "groesste": float(dev.max())}
    # S4 gefahrene Kosten ab x = 10 (Faktor 1/2)
    A, B, Q = D.plant("skalar")
    Pd, Kd = D.dare(A, B, Q, 1.0)
    x0 = np.array([10.0])

    def kosten(law):
        return 0.5 * D.run_loop(law, x0, A, B, Q, 1.0, T=600, tol=1e-10).cost

    c_mpc2 = kosten(D.MPCLaw(D.build_mpc("skalar", 1.0, 2, 1.0, 1.0, "Q")))
    c_clip = kosten(D.ClipLaw(Kd, 1.0))
    c_mpc40 = kosten(D.MPCLaw(D.build_mpc("skalar", 1.0, 40, 1.0, 1.0, "DARE")))
    h = 0.005
    xg = np.arange(-14.0, 14.0 + 1e-9, h)
    ug = np.arange(-1.0, 1.0 + 1e-9, h)
    V = 0.5 * float(Pd[0, 0]) * xg ** 2
    sweeps = 0
    for sweeps in range(1, 2001):
        xn = np.clip(xg[:, None] + ug[None, :], xg[0], xg[-1])
        Vn = np.interp(xn.ravel(), xg, V).reshape(xn.shape)
        Vneu = np.min(0.5 * (xg[:, None] ** 2 + ug[None, :] ** 2) + Vn, axis=1)
        d = float(np.max(np.abs(Vneu - V)))
        V = Vneu
        if d < 1e-10:
            break
    out["s4"] = {"mpc_n2": c_mpc2, "clip": c_clip, "mpc_n40": c_mpc40, "dp_gitter": float(V[int(round((10.0 - xg[0]) / h))]), "dp_schritt": h, "dp_sweeps": sweeps,
                 "gewinn_dare": float(Kd[0, 0])}
    save("skalar", out)


# ---------------------------------------------------------------------------------------------------------------- gegenprobe
def teil_gegenprobe():
    rng = SplitMix64(SEED)
    n_inst = 1000
    T = 3000
    teurer = {2: 0, 3: 0, 5: 0}
    gleich = {2: 0, 3: 0, 5: 0}
    billiger = {2: 0, 3: 0, 5: 0}
    vorteil = {2: 0.0, 3: 0.0, 5: 0.0}
    nachteil = {2: 0.0, 3: 0.0, 5: 0.0}
    nkonv = 0
    for _ in range(n_inst):
        R = (0.1, 1.0, 10.0)[rng.below(3)]
        umax = 0.2 + 1.8 * rng.random()
        x0 = np.array([-30.0 + 60.0 * rng.random(), -12.0 + 24.0 * rng.random()])
        prob = D.build_mpc("wagen", 1.0, 2, R, umax, "DARE")
        lc = D.run_loop(D.ClipLaw(prob.K, umax), x0, prob.A, prob.B, prob.Q, R, T)
        if lc.end_norm > C.CONV_COUNT_TOL:
            continue
        nkonv += 1
        for N in (2, 3, 5):
            pm = D.build_mpc("wagen", 1.0, N, R, umax, "DARE")
            lm = D.run_loop(D.MPCLaw(pm), x0, pm.A, pm.B, pm.Q, R, T)
            assert lm.end_norm <= C.CONV_COUNT_TOL, "MPC konvergiert nicht"
            rel = lm.cost / lc.cost - 1
            if rel > 1e-9:
                teurer[N] += 1
                nachteil[N] = max(nachteil[N], rel)
            elif rel < -1e-9:
                billiger[N] += 1
                vorteil[N] = max(vorteil[N], -rel)
            else:
                gleich[N] += 1
    save("gegenprobe", {"instanzen": n_inst, "clip_konvergiert": nkonv, "schritte": T, "seed": SEED,
                        "n": {str(N): {"teurer": teurer[N], "gleich": gleich[N], "billiger": billiger[N], "vorteil_max_prozent": 100 * vorteil[N], "nachteil_max_prozent": 100 * nachteil[N]}
                              for N in (2, 3, 5)}})


# ---------------------------------------------------------------------------------------------------------------- weit
def teil_weit():
    out = {"gitter": {}, "zufall": {}, "mpc": {}}
    pts = np.array([(a, b) for a in np.arange(-300.0, 300.0 + 1e-9, 30.0) for b in np.arange(-60.0, 60.0 + 1e-9, 6.0) if (a, b) != (0.0, 0.0)])
    for R in (1.0, 0.1, 0.01):
        A, B, Q = D.plant("wagen")
        _, K = D.dare(A, B, Q, R)
        _, end, gmax, _ = D.run_clip_batch(pts, K, 1.0, A, B, Q, R, T=3000, tol=1e-9)
        out["gitter"][f"{R:g}"] = {"punkte": len(pts), "nicht_konvergiert": int(np.sum(end > C.CONV_COUNT_TOL)), "max_verhaeltnis": float(np.max(gmax / np.linalg.norm(pts, axis=1)))}
    rng = SplitMix64(SEED)
    n = 500
    st = np.array([[-1000.0 + 2000.0 * rng.random(), -200.0 + 400.0 * rng.random()] for _ in range(n)])
    A, B, Q = D.plant("wagen")
    _, K = D.dare(A, B, Q, 1.0)
    cost, end, gmax, steps = D.run_clip_batch(st, K, 1.0, A, B, Q, 1.0, T=200000, tol=1e-6)
    ratio = gmax / np.linalg.norm(st, axis=1)
    out["zufall"] = {"starts": n, "nicht_konvergiert": int(np.sum(end >= 1e-6)), "schritte_median": float(np.median(steps)), "schritte_max": int(steps.max()),
                     "verhaeltnis_median": float(np.median(ratio)), "verhaeltnis_max": float(ratio.max())}
    # MPC N = 10 an den ersten 40 dieser Starts
    prob = D.build_mpc("wagen", 1.0, 10, 1.0, 1.0, "DARE")
    m = 40
    conv, sm, rm, cr = 0, [], [], []
    for i in range(m):
        lp = D.run_loop(D.MPCLaw(prob), st[i], prob.A, prob.B, prob.Q, 1.0, T=20000, tol=1e-6)
        conv += int(lp.end_norm < 1e-6)
        sm.append(lp.steps)
        rm.append(lp.max_norm / float(np.linalg.norm(st[i])))
        cr.append(lp.cost / float(cost[i]))
    out["mpc"] = {"starts": m, "konvergiert": conv, "schritte_median": float(np.median(sm)), "schritte_max": int(max(sm)), "verhaeltnis_median": float(np.median(rm)),
                  "verhaeltnis_max": float(max(rm)), "kosten_vs_clip_min": float(min(cr)), "kosten_vs_clip_max": float(max(cr)), "N": 10}
    save("weit", out)


# ---------------------------------------------------------------------------------------------------------------- instabil
def teil_instabil():
    pts = D.grid_points(C.UNSTABLE_GRID, C.UNSTABLE_GRID)
    out = {"punkte": len(pts), "gitter": list(C.UNSTABLE_GRID), "a": {}}
    for a in C.A_CHOICES:
        t0 = time.time()
        A, B, Q = D.plant("wagen", a)
        _, K = D.dare(A, B, Q, C.UNSTABLE_R)
        clip = D.ClipLaw(K, C.UNSTABLE_UMAX)
        e = {"clip": [int(D.converges(clip, p, A, B)) for p in pts], "mpc": {}}
        for N in C.E6_N_CHOICES:
            law = D.MPCLaw(D.build_mpc("wagen", a, N, C.UNSTABLE_R, C.UNSTABLE_UMAX, "DARE"))
            e["mpc"][str(N)] = [int(D.converges(law, p, A, B)) for p in pts]
        e["steuerbar"] = [int(v) for v in D.controllable_mask(A, B, pts)]
        out["a"][f"{a:g}"] = e
        print("instabil", a, f"{time.time() - t0:.0f} s", flush=True)
    save("instabil", out)


# ---------------------------------------------------------------------------------------------------------------- orakel
def enumerate_box_qp(H, f, lo, hi):
    """Alle 3^N Muster in Gleitkomma (anderer Rechenweg als die Aktivmengen-Methode)."""
    N = len(f)
    best, bestv = None, np.inf
    for pat in itertools.product((-1, 0, 1), repeat=N):
        pat = np.array(pat)
        u = np.where(pat < 0, lo, np.where(pat > 0, hi, 0.0)).astype(float)
        free = pat == 0
        if free.any():
            u[free] = np.linalg.solve(H[np.ix_(free, free)], -(f[free] + H[np.ix_(free, ~free)] @ u[~free]))
        if np.all(u >= lo - 1e-12) and np.all(u <= hi + 1e-12):
            v = 0.5 * u @ H @ u + f @ u
            if v < bestv:
                best, bestv = u.copy(), v
    return best


def slsqp(H, f, lo, hi):
    from scipy.optimize import minimize

    return minimize(lambda u: 0.5 * u @ H @ u + f @ u, np.clip(np.zeros(len(f)), lo, hi), jac=lambda u: H @ u + f, bounds=list(zip(lo, hi)), method="SLSQP",
                    options={"ftol": 1e-15, "maxiter": 1000}).x


def teil_orakel():
    rng = SplitMix64(SEED)
    dev = {"Aufzählung": 0.0, "bvls": 0.0, "SLSQP": 0.0}
    n_inst = 400
    with_active = 0
    for _ in range(n_inst):
        N = 1 + rng.below(7)
        M = np.array([[rng.normal() for _ in range(N)] for _ in range(N)])
        H = M @ M.T + 0.1 * np.eye(N)
        f = np.array([3.0 * rng.normal() for _ in range(N)])
        lo = -np.array([0.2 + 1.8 * rng.random() for _ in range(N)])
        hi = np.array([0.2 + 1.8 * rng.random() for _ in range(N)])
        u = QP.qp_active_set(H, f, lo, hi).u
        with_active += int(np.any(np.isclose(u, lo) | np.isclose(u, hi)))
        for name, fn in (("Aufzählung", enumerate_box_qp), ("bvls", QP.qp_bvls), ("SLSQP", slsqp)):
            dev[name] = max(dev[name], float(np.max(np.abs(u - fn(H, f, lo, hi)))))
    out = {"zufalls_qps": {"anzahl": n_inst, "mit_aktiver_grenze": with_active, "abweichung": dev}}
    # MPC-QPs entlang des Wagens
    prob_dev = {"Aufzählung": 0.0, "bvls": 0.0, "SLSQP": 0.0}
    nq = 0
    for N in (2, 3, 4, 5, 6, 7):
        prob = D.build_mpc("wagen", 1.0, N, 1.0, 1.0, "DARE")
        for _ in range(40):
            x0 = np.array([-10.0 + 20.0 * rng.random(), -4.0 + 8.0 * rng.random()])
            f = prob.F @ x0
            u = QP.qp_active_set(prob.H, f, prob.lo, prob.hi).u
            for name, fn in (("Aufzählung", enumerate_box_qp), ("bvls", QP.qp_bvls), ("SLSQP", slsqp)):
                prob_dev[name] = max(prob_dev[name], float(np.max(np.abs(u - fn(prob.H, f, prob.lo, prob.hi)))))
            nq += 1
    out["mpc_qps"] = {"anzahl": nq, "abweichung": prob_dev}
    # große N: Wert-Überschuss von bvls und SLSQP gegen die eigene Methode, KKT-Verletzung
    gross = {}
    for N in (10, 20, 30, 60):
        prob = D.build_mpc("wagen", 1.0, N, 1.0, 1.0, "DARE")
        dv_b, dv_s, kkt, nact = 0.0, 0.0, 0.0, 0
        for _ in range(30):
            x0 = np.array([-10.0 + 20.0 * rng.random(), -4.0 + 8.0 * rng.random()])
            f = prob.F @ x0
            u = QP.qp_solve(prob.H, f, prob.lo, prob.hi).u
            g = prob.H @ u + f
            innen = np.abs(u) < 1 - 1e-9
            kkt = max(kkt, float(np.max(np.abs(g[innen]), initial=0.0)), float(np.max(np.maximum(0.0, -g[u <= -1 + 1e-9]), initial=0.0)),
                      float(np.max(np.maximum(0.0, g[u >= 1 - 1e-9]), initial=0.0)))
            nact += int(np.any(np.abs(u) > 1 - 1e-9))
            val = lambda v: 0.5 * v @ prob.H @ v + f @ v  # noqa: E731
            dv_b = max(dv_b, val(QP.qp_bvls(prob.H, f, prob.lo, prob.hi)) - val(u))
            dv_s = max(dv_s, val(slsqp(prob.H, f, prob.lo, prob.hi)) - val(u))
        gross[str(N)] = {"cond": float(np.linalg.cond(prob.H)), "wert_ueberschuss_bvls": dv_b, "wert_ueberschuss_slsqp": dv_s, "kkt_max": kkt, "mit_aktiver_grenze": nact}
    out["gross"] = gross
    # Stellfolgen-Gitter N = 3
    prob = D.build_mpc("wagen", 1.0, 3, 1.0, 1.0, "DARE")
    g = np.linspace(-1, 1, 41)
    U = np.array(list(itertools.product(g, g, g)))
    gitter = []
    for x0 in ([10.0, 0.0], [-6.0, 2.0], [1.0, 0.5]):
        f = prob.F @ np.array(x0)
        vals = 0.5 * np.einsum("ij,jk,ik->i", U, prob.H, U) + U @ f
        u = QP.qp_active_set(prob.H, f, prob.lo, prob.hi).u
        vq = 0.5 * u @ prob.H @ u + f @ u
        gitter.append({"x0": x0, "abstand": float(vals.min() - vq), "nie_darunter": bool(vals.min() >= vq - 1e-12)})
    out["gitter"] = {"folgen": len(U), "faelle": gitter}
    save("orakel", out)


# ---------------------------------------------------------------------------------------------------------------- rueckfall
def teil_rueckfall():
    out = {}
    pts = D.grid_points(C.UNSTABLE_GRID, C.UNSTABLE_GRID)
    for a in (1.0, 1.2):
        A, B, Q = D.plant("wagen", a)
        for N in (3, 5, 10, 20, 40):
            t0 = time.time()
            prob = D.build_mpc("wagen", a, N, 1.0, 1.0, "DARE")
            law = D.MPCLaw(prob, record=True)
            for p in pts:
                x = np.array(p)
                for _ in range(C.STEPS_MAX):
                    x = A @ x + B[:, 0] * law(x)
                    nrm = np.linalg.norm(x)
                    if nrm < 1e-6 or nrm > 1e6:
                        break
            out[f"{a:g}|{N}"] = {"cond": float(np.linalg.cond(prob.H)), "aufrufe": law.calls, "rueckfaelle": law.fallbacks}
            print("rueckfall", a, N, out[f"{a:g}|{N}"], f"{time.time() - t0:.0f} s", flush=True)
    save("rueckfall", out)


TEILE = {"kurz": teil_kurz, "d1": teil_d1, "skalar": teil_skalar, "gegenprobe": teil_gegenprobe, "weit": teil_weit, "instabil": teil_instabil, "orakel": teil_orakel,
         "rueckfall": teil_rueckfall}

if __name__ == "__main__":
    wahl = sys.argv[1:] or ["alle"]
    if wahl == ["alle"]:
        wahl = list(TEILE)
    for t in wahl:
        t0 = time.time()
        TEILE[t]()
        print(f"{t}: {time.time() - t0:.0f} s")
