"""Stellgrenzen: MPC als quadratisches Programm - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Wagen (Doppelintegrator) und ein Skalar mit Stellgrenze |u| <= u_max: MPC löst in jedem Takt ein quadratisches Programm (QP) mit Boxgrenzen und wird mit
„LQR, dann abschneiden“ verglichen. Sechs Experimente; Aufbau nach DEMO-PLAYBOOK §2 (Schnellstart und allgemeine Einstellungen in der Seitenleiste).

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import numpy as np
import streamlit as st

import msg_constants as C
import msg_dynamics as D
import msg_evaluation as EV
from msg_evaluation import de, frac_text as ft, sci
from msg_layout import dunkles_design, kopf_css
from msg_presets import (
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)
from msg_visualization import (
    build_distance,
    build_hand_contours,
    build_hand_cost,
    build_horizon,
    build_scaling,
    build_sequence,
    build_skalar_law,
    build_stability_map,
    build_trajectory,
    build_wagen_map,
)

st.set_page_config(page_title="Stellgrenzen – Sebastian Hanisch", layout="wide")


# ---------------------------------------------------------------------------------------------------------------- zwischengespeicherte Rechnungen
# Schlüssel sind nur die Eingaben, die eine Rechnung wirklich braucht; sonst würde jeder unbeteiligte Regler das Gitter neu rechnen.
def _s(n, umax, r, terminal, x1=C.DEFAULT_X1, x2=C.DEFAULT_X2):
    return EV.Settings(n=int(n), umax=float(umax), r=float(r), terminal=terminal, x1=float(x1), x2=float(x2))


@st.cache_data(show_spinner=False)
def _hand(x):
    return EV.hand(Fraction(x))


@st.cache_data(show_spinner="Rechnet die 356 Startzustände (je nach Horizont etwa 1 bis 5 Sekunden) …")
def _grid(n, umax, r, terminal):
    return EV.wagen_grid(_s(n, umax, r, terminal))


@st.cache_data(show_spinner=False)
def _firstu(n, umax, r, terminal):
    return EV.firstu_check(_s(n, umax, r, terminal))


@st.cache_data(show_spinner=False)
def _skalar(n, umax, r, terminal, x1):
    return EV.skalar_firstu(_s(n, umax, r, terminal, x1=x1))


@st.cache_data(show_spinner=False)
def _scaling(n, umax, r, terminal, x1, x2):
    return EV.scaling_check(_s(n, umax, r, terminal, x1, x2))


@st.cache_data(show_spinner=False)
def _prediction(n, umax, r, terminal, x1, x2):
    return EV.prediction(_s(n, umax, r, terminal, x1, x2))


@st.cache_data(show_spinner=False)
def _trace(n, umax, r, terminal, x1, x2):
    return EV.qp_trace(_s(n, umax, r, terminal, x1, x2))


@st.cache_data(show_spinner=False)
def _traj(n, umax, r, terminal, x1, x2):
    return EV.trajectory(_s(n, umax, r, terminal, x1, x2))


# ---------------------------------------------------------------------------------------------------------------- Kopf und Einleitung
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Stellgrenzen: MPC als quadratisches Programm")
st.markdown(
    """
Das Gesetz **u = −K x** der Riccati-Gleichung setzt eine **unbegrenzte Stellgröße** voraus. Am Wagen (Doppelintegrator) aus dem Startzustand (10; 0) verlangt es u = −4,22, die Grenze ist 1.
Mit Grenze löst **MPC** in jedem Takt ein **quadratisches Programm (QP) mit Boxgrenzen** und wendet das erste Stellsignal an; die Gegenspielerin ist **„LQR, dann abschneiden“**
(u = clip(−K x)). Die Demo fragt, wo das QP etwas kann, was das Abschneiden nicht kann, und wo nicht: im Skalar und bei Horizont 1 gar nicht, am Wagen ab Horizont 2, sofern die Grenze auf dem Gitter greift (u_max = 1, R = 1: bei N = 2 in 142 von 356 Startzuständen).

Weiter unten: sechs **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Alle Zahlen gelten in diesem Modell, bei den eingestellten Parametern.
"""
)

with st.expander("So funktioniert MPC mit Stellgrenze", expanded=True):
    st.markdown(
        """
1. **Modell:** Zustand x, Stellsignal u, x⁺ = A x + B u. Kosten über N Takte: Σ (x'Qx + R·u²) plus Endkosten x_N'P x_N. Die Stellgrenze lautet −u_max ≤ u ≤ u_max.
2. **Das QP:** Setzt man die Zustände ein, bleibt eine Zielfunktion ½ u'Hu + f'u in der Stellfolge u = (u₀ … u_{N−1}). H ist fest, nur f = F·x₀ wechselt mit dem Zustand, die Nebenbedingung ist die Box.
3. **Anwenden und neu lösen:** Nur u₀ wird gefahren; im nächsten Takt wird mit dem neuen Zustand neu gelöst.
4. **Gegenspielerin:** „LQR, dann abschneiden“ rechnet u = −K x mit dem Gewinn K ohne Grenze und schneidet das Ergebnis bei ±u_max ab. Das QP weiß dagegen, dass spätere Stellsignale an der Grenze liegen werden, und passt das erste an.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# ---------------------------------------------------------------------------------------------------------------- Seitenleiste
with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Regler (Wagen)**")
    n = st.slider("Horizont N", C.N_MIN, C.N_MAX, key="n_slider", help="Wie viele Takte das QP vorausplant (höchstens 20: darüber wird H schlecht konditioniert).")
    umax = st.select_slider("Stellgrenze u_max", options=C.UMAX_CHOICES, key="umax_slider", format_func=lambda v: de(v, 2),
                            help="Nur diese Werte, weil die Referenzkosten für sie vorgerechnet sind.")
    r = st.select_slider("Stellaufwand R", options=C.R_CHOICES, key="r_slider", format_func=lambda v: de(v, 1), help="Gewicht von u² in den Kosten.")
    terminal = st.radio("Endkosten", C.TERMINALS, key="terminal", format_func=lambda k: C.TERMINAL_LABELS[k],
                        help="DARE: Riccati-Lösung, der Gewinn ohne Grenze ist für jeden Horizont derselbe wie der unendliche. Q: nur x'Qx am Ende.")
    st.markdown("**Startzustand**")
    x1 = st.slider("Ort x₁", C.X1_MIN, C.X1_MAX, step=C.X1_STEP, key="x1_slider", help="Auch der Startwert des Skalars in Experiment 2.")
    x2 = st.slider("Geschwindigkeit x₂", C.X2_MIN, C.X2_MAX, step=C.X2_STEP, key="x2_slider")

system = st.session_state["e2_system"]
e1_x = st.session_state["e1_x"]
e6_a = st.session_state["e6_a"]
e6_n = st.session_state["e6_n"]
sync_query_params({"n_slider": int(n), "umax_slider": umax, "r_slider": r, "terminal": terminal, "x1_slider": x1, "x2_slider": x2,
                   "e1_x": e1_x, "e2_system": system, "e6_a": e6_a, "e6_n": e6_n})

args = (int(n), float(umax), float(r), terminal)
settings = EV.Settings(int(n), float(umax), float(r), terminal, float(x1), float(x2), float(e1_x), system, float(e6_a), int(e6_n))

# ---------------------------------------------------------------------------------------------------------------- Ergebnis
st.markdown("## 🎯 Ergebnis: ein Startzustand, zwei Regeln")
traj = _traj(*args, float(x1), float(x2))
m1, m2, m3 = st.columns(3)
mpc_lp, clip_lp = traj["mpc"], traj["clip"]
ok = mpc_lp.end_norm < C.CONV_COUNT_TOL and clip_lp.end_norm < C.CONV_COUNT_TOL
# Delta immer als „mein Wert minus Referenz“ (MPC minus Abschneiden); die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Kosten MPC", de(mpc_lp.cost, 2) if mpc_lp.end_norm < C.CONV_COUNT_TOL else "nicht konvergiert",
          delta=f"{mpc_lp.cost - clip_lp.cost:+.2f} gegen Abschneiden".replace(".", ",") if ok else None, delta_color="inverse")
m2.metric("Kosten LQR, dann abschneiden", de(clip_lp.cost, 2) if clip_lp.end_norm < C.CONV_COUNT_TOL else "nicht konvergiert", delta="Referenz", delta_color="off")
m3.metric("Takte bis |x| < 1e-9", f"{mpc_lp.steps} (MPC) / {clip_lp.steps}", delta="MPC / Abschneiden", delta_color="off")
st.plotly_chart(build_trajectory(traj), width="stretch", key="main_chart")
st.caption("Gestrichelte graue Linien: die Stellgrenzen ±u_max. Gefahren wird bis zur Norm 1e-9 (höchstens 400 Takte); Kosten Σ(x'Qx + R·u²) ohne Faktor ½.")
kind, text = EV.verdict_start(traj, settings)
getattr(st, kind)(text)

# ---------------------------------------------------------------------------------------------------------------- Kernfragen
grid = _grid(*args)
counts = EV.wagen_counts(grid)
firstu = _firstu(*args)
tr = _trace(*args, float(x1), float(x2))
pred = _prediction(*args, float(x1), float(x2))
hstats = EV.horizon_stats(settings, grid)

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Was ist hier ein QP, und wie wird es gelöst?** Eine Zielfunktion zweiten Grades in der Stellfolge mit Boxgrenzen; die Aktivmengen-Methode (Experiment 5) legt fest, welche Grenzen aktiv sind.")
kind, text = EV.verdict_qp(tr, settings)
getattr(st, kind)(text)

st.markdown("**2. Wann reicht „LQR, dann abschneiden“, wann nicht?** Der Vergleich der gefahrenen Kosten (Experiment 2 und 3).")
if system == "Skalar":
    sk = _skalar(*args, float(x1))
    kind, text = EV.verdict_skalar(sk, settings)
else:
    kind, text = EV.verdict_wagen(counts, settings, firstu)
getattr(st, kind)(text)

st.markdown("**3. Wie lang muss der Horizont sein?** Der Aufschlag gegen die Referenz (N = 60) über das Gitter (Experiment 4).")
kind, text = EV.verdict_horizon(settings, hstats, pred)
getattr(st, kind)(text)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Experiment 1
st.subheader("🔬 Experiment 1: Von Hand, das QP des Skalars")
st.markdown(
    "Der Skalar x⁺ = x + u mit Stufenkosten ½(x² + u²), Endkosten ½x², Horizont N = 2 und Grenze |u| ≤ 1 (Beispiel aus Rawlings/Mayne/Diehl, Kap. 2; hier mit Brüchen nachgerechnet; "
    "Faktor ½ nur in diesem Experiment). Links die Höhenlinien der Kosten über der Ebene der beiden Stellsignale mit der Box, rechts die gefahrene Kostensumme gegen die Voraussage."
)
e1 = st.slider("Startwert x", C.E1_MIN, C.E1_MAX, step=C.E1_STEP, key="e1_x", help="Im Bereich |x| ≤ 5/3 greift keine Grenze, bis |x| = 3 nur u₀, darüber beide.")
h = _hand(float(e1))
c1, c2 = st.columns(2)
c1.plotly_chart(build_hand_contours(float(e1), [float(u) for u in h["u"]]), width="stretch", key="e1_contour")
c2.plotly_chart(build_hand_cost(EV.hand_cost_path(h), float(h["V"]), float(h["gefahren"])), width="stretch", key="e1_cost")
kind, text = EV.verdict_hand(h)
getattr(st, kind)(text)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Experiment 2
st.subheader("🔬 Experiment 2: Abschneiden gegen MPC")
st.radio("System", C.SYSTEMS, key="e2_system", horizontal=True,
         help="Wagen: Karte der Mehrkosten über 356 Startzustände. Skalar: das erste Stellsignal auf 2 401 Punkten (Startwert x₁ der Seitenleiste für die Kosten).")
if system == "Skalar":
    st.markdown("Skalar: erstes Stellsignal des QP gegen das erste Element der abgeschnittenen unbeschränkten N-Schritt-Lösung. Die Linien liegen übereinander. "
                "Das Urteil steht oben unter Kernfrage 2.")
    st.plotly_chart(build_skalar_law(sk), width="stretch", key="e2_skalar")
else:
    st.markdown("Wagen: relative Mehrkosten von „LQR, dann abschneiden“ gegen MPC (gefahrene Kosten bis |x| < 1e-9) für jeden der 356 Startzustände "
                "(x₁ von −10 bis 10, x₂ von −4 bis 4). Weiß heißt gleiche Kosten. Das Urteil steht oben unter Kernfrage 2. „Nie teurer“ gilt nur für dieses Gitter und die Zufallsinstanzen der Gegenprobe und nur mit Endkosten DARE. " + EV.gegenprobe_text())
    mk = EV.wagen_mehrkosten(grid)
    st.plotly_chart(build_wagen_map(grid.pts, np.where(np.isfinite(mk), mk, np.nan), D.grid_axis(C.GRID_X1), D.grid_axis(C.GRID_X2)), width="stretch", key="e2_wagen")
    st.caption(f"Mittlere Iterationszahl der Aktivmengen-Methode über diese Läufe: {de(grid.iterations_mean, 2)}; Rückfälle auf den zweiten Löser: {grid.fallbacks}.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Experiment 3
st.subheader("🔬 Experiment 3: Es zählt nur das Verhältnis")
st.markdown("Vervielfacht man Startzustand **und** Grenze mit demselben Faktor c, vervielfachen sich die Kosten mit c². Links der Beweis am Startzustand der Seitenleiste, rechts die Mehrkosten des Abschneidens "
            "über dem Abstand ‖x₀‖/u_max aller Gitterpunkte (Einstellungen der Seitenleiste).")
sc = _scaling(*args, float(x1), float(x2))
ab = EV.abstand_aufschlag(grid, float(umax))
d1, d2 = st.columns(2)
d1.plotly_chart(build_scaling(sc), width="stretch", key="e3_scaling")
d2.plotly_chart(build_distance(ab), width="stretch", key="e3_distance")
kind, text = EV.verdict_scaling(sc, ab, settings)
getattr(st, kind)(text)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Experiment 4
st.subheader("🔬 Experiment 4: Horizont und Endkosten")
st.markdown("Der größte Aufschlag von MPC gegen die Referenz (N = 60, Endkosten DARE) über die 356 Startzustände, je Horizont (**vorgerechnet**, Terminalkosten DARE). "
            "Die gestrichelte Linie markiert den Horizont der Seitenleiste; das Urteil für die eingestellte Kombination steht oben unter Kernfrage 3.")
fz = EV.frozen().get("d1")
if fz:
    st.plotly_chart(build_horizon(fz["szenarien"], EV.ref_key(float(r), float(umax)), int(n), terminal), width="stretch", key="e4_horizon")
    key = EV.ref_key(float(r), float(umax))
    sc_tab = fz["szenarien"].get(key)
    if sc_tab is None:
        key, sc_tab = EV.ref_key(1.0, 1.0), fz["szenarien"][EV.ref_key(1.0, 1.0)]
        st.caption("Für die eingestellte Kombination aus u_max und R liegt keine Tabelle vor; gezeigt ist u_max = 1, R = 1.")
    rows = [{"Horizont N": int(nn), "konvergiert (von 356)": e["konvergiert"], "Aufschlag Median (%)": e["median"], "Aufschlag Maximum (%)": e["max"], "MPC billiger": e["billiger"],
             "gleich": e["gleich"], "teurer": e["teurer"]} for nn, e in sorted(sc_tab["mpc"].items(), key=lambda kv: int(kv[0]))]
    cfg = {"Aufschlag Median (%)": st.column_config.NumberColumn(format="%.3f %%"), "Aufschlag Maximum (%)": st.column_config.NumberColumn(format="%.3f %%")}
    st.markdown(f"**Vorgerechnet, Endkosten DARE, Kombination (R = {key.split('|')[0].replace('.', ',')}; u_max = {key.split('|')[1].replace('.', ',')}):**")
    st.dataframe(rows, hide_index=True, column_config=cfg, width="stretch")
    if terminal == "Q" and fz.get("d6"):
        st.markdown("**Ohne Endkosten (P = Q), vorgerechnet, u_max = 1, R = 1:** MPC gegen die Referenz und gegen Abschneiden.")
        rows6 = [{"Horizont N": int(nn), "konvergiert (von 356)": e["konvergiert"], "Aufschlag Maximum (%)": e["max"], "MPC billiger": e["billiger"], "teurer": e["teurer"]}
                 for nn, e in sorted(fz["d6"].items(), key=lambda kv: int(kv[0]))]
        st.dataframe(rows6, hide_index=True, column_config={"Aufschlag Maximum (%)": st.column_config.NumberColumn(format="%.2f %%")}, width="stretch")
else:
    st.info("Die vorgerechneten Messreihen fehlen (data/msg_frozen_d1.json).")
st.markdown("**Voraussage gegen gefahren** (Startzustand der Seitenleiste): Der vorhergesagte Wert V_N(x₀) des QP ist der Wert über N Takte plus Endkosten; gefahren werden die Kosten bis ins Unendliche.")
p1, p2, p3, p4 = st.columns(4)
p1.metric(f"V (N = {pred['n']['N']}) vorhergesagt", de(pred["n"]["V"], 2))
p2.metric(f"gefahren (N = {pred['n']['N']})", de(pred["n"]["gefahren"], 2))
p3.metric("V (N = 60) vorhergesagt", de(pred["ref"]["V"], 2))
p4.metric("gefahren (N = 60)", de(pred["ref"]["gefahren"], 2))

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Experiment 5
st.subheader("🔬 Experiment 5: Das QP anschauen")
st.markdown("Aktivmengen-Methode, Schritt für Schritt, für das QP im Startzustand der Seitenleiste (Kaltstart bei u = 0). „unten fest“ und „oben fest“ sind die Stellsignale im Arbeitsset; "
            "die Zielwerte sind der Wert V des Horizontproblems (mit Konstante). Das Urteil steht oben unter Kernfrage 1.")
rows5 = EV.trace_table(tr)
st.dataframe(rows5, hide_index=True, column_config={"Zielwert V": st.column_config.NumberColumn(format="%.4f")}, width="stretch")
st.plotly_chart(build_sequence(tr), width="stretch", key="e5_sequence")

with st.expander("Von Hand: Wagen, N = 2, u_max = ½, Start (7; −2), Endkosten P = I"):
    wh = EV.wagen_hand()
    st.markdown(
        f"Unbeschränkte Lösung u = ({ft(wh['unbeschraenkt'][0])}; {ft(wh['unbeschraenkt'][1])}), ihr Abschneiden ergibt ({ft(wh['abgeschnitten'][0])}; {ft(wh['abgeschnitten'][1])}). "
        f"Das QP mit Box liefert ({ft(wh['mpc'][0])}; {ft(wh['mpc'][1])}): liegt u₁ an der Grenze, fehlt später Bremswirkung, und u₀ muss jetzt etwas tun. "
        f"Kosten: unbeschränkt (nicht zulässig) {ft(wh['J_unbeschraenkt'])}, MPC {ft(wh['J_mpc'])} = {de(float(wh['J_mpc']), 4)}, abgeschnittene Folge {ft(wh['J_abgeschnitten'])}. "
        f"Gradient g = H u + f = ({ft(wh['gradient'][0])}; {ft(wh['gradient'][1])}): u₀ frei, u₁ oben mit g₁ < 0. Alles mit Brüchen gerechnet (`msg_exact.py`)."
    )

orakel = EV.frozen().get("orakel")
if orakel:
    with st.expander("Orakel: das QP gegen drei andere Rechenwege (vorgerechnet)"):
        zq, mq = orakel["zufalls_qps"], orakel["mpc_qps"]
        st.markdown("Größte Abweichung der Lösung: eigene Methode gegen Aufzählung aller 3^N Muster, `bvls` (beschränkte kleinste Quadrate) und SLSQP.")
        st.dataframe([{"Prüfmenge": f"{zq['anzahl']} Zufalls-QPs (N = 1 bis 7)", **{k: f"{sci(v)}" for k, v in zq["abweichung"].items()}},
                      {"Prüfmenge": f"{mq['anzahl']} MPC-QPs entlang des Wagens (N = 2 bis 7)", **{k: f"{sci(v)}" for k, v in mq["abweichung"].items()}}], hide_index=True, width="stretch")
        gr = orakel["gross"]
        st.markdown("Große N (je 30 Zufallszustände): um wie viel liegt der Zielwert von `bvls` und SLSQP über dem der eigenen Methode; größte Verletzung der Optimalitätsbedingungen (KKT) der eigenen Lösung.")
        st.dataframe([{"N": int(k), "Konditionszahl H": f"{sci(v['cond'])}", "bvls": f"{sci(v['wert_ueberschuss_bvls'])}", "SLSQP": f"{sci(v['wert_ueberschuss_slsqp'])}",
                       "KKT-Verletzung": f"{sci(v['kkt_max'])}"} for k, v in gr.items()], hide_index=True, width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Experiment 6
st.subheader("🔬 Experiment 6: Offen instabile Strecke")
st.markdown("Wagen mit A = [[a, 1], [0, 1]]: für a > 1 wächst der Ort ohne Eingriff exponentiell. Gitter von 624 Startzuständen in [−6, 6]², Grenze 1, R = 1, Endkosten DARE; "
            "grün konvergiert, rot nicht; die schwarze Umrisslinie ist die Menge der Startzustände, die sich per linearem Programm in 100 Schritten mit |u| ≤ 1 auf den Ursprung steuern lassen. "
            "**Vorgerechnet**.")
e6c1, e6c2 = st.columns(2)
e6c1.select_slider("Instabilität a", options=C.A_CHOICES, key="e6_a", format_func=lambda v: de(v, 2))
e6c2.select_slider("Horizont N", options=C.E6_N_CHOICES, key="e6_n")
mp = EV.instabil_maps(float(st.session_state["e6_a"]), int(st.session_state["e6_n"]))
if mp is not None:
    f1, f2 = st.columns(2)
    f1.plotly_chart(build_stability_map(mp, "clip", f"LQR, dann abschneiden: {mp['clip_n']} von {mp['punkte']}"), width="stretch", key="e6_clip")
    f2.plotly_chart(build_stability_map(mp, "mpc", f"MPC, N = {int(st.session_state['e6_n'])}: {mp['mpc_n']} von {mp['punkte']}"), width="stretch", key="e6_mpc")
kind, text = EV.verdict_instabil(mp, float(st.session_state["e6_a"]), int(st.session_state["e6_n"]))
getattr(st, kind)(text)
st.caption("Hinweis: Ab N = 40 und a = 1,2 bricht die eigene Aktivmengen-Methode in einem Teil der Aufrufe ab und braucht den Rückfall (siehe unten); deshalb ist der Horizont hier und in der Seitenleiste auf 20 begrenzt.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Grenzen
st.subheader("🚧 Wo die Annahmen enden")
tabelle = "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n|---|---|---|\n" + "\n".join(f"| **{a}** | {b} | {c} |" for a, b, c in EV.limits_rows())
st.markdown(tabelle)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Formeln
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Strecke $x^+ = Ax + Bu$ mit einem Eingang, die Gewichte $Q = I$ und $R > 0$, die Endkosten $P$, ein Horizont $N$, eine Grenze $u_{\max}$ und ein Startzustand $x_0$. Gesucht ist die Stellfolge
$u = (u_0, \dots, u_{N-1})$, die die Kosten kleinstmöglich macht; angewendet wird nur $u_0$.

| Symbol | Bedeutung |
|---|---|
| $x_k$ | Zustand im Takt $k$ (Wagen: Ort und Geschwindigkeit) |
| $u_k$ | Stellsignal im Takt $k$, $\lvert u_k \rvert \le u_{\max}$ |
| $A, B$ | Wagen: $A = \begin{pmatrix} a & 1 \\ 0 & 1 \end{pmatrix}$, $B = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$; Skalar: $A = B = 1$ |
| $Q, R, P$ | Gewichte der Zustände, der Stellgröße und der Endzustände |
| $\Phi, \Gamma$ | $x_k = A^k x_0 + \sum_{j<k} A^{k-1-j} B u_j$ als Matrizen (Zustände aller Takte gestapelt) |
| $H, F$ | Matrizen des QP, $H$ fest, $f = F x_0$ wechselt mit dem Zustand |
| $K, P_\infty$ | Gewinn und Lösung der Riccati-Gleichung (DARE) |

**Kosten.** Ohne Faktor $\tfrac12$ (im Skalar-Handbeispiel mit ½, wie bei Rawlings, damit 123,5 vergleichbar bleibt):

$$J(x_0, u) = x_0^\top Q x_0 + \sum_{k=1}^{N-1} x_k^\top Q x_k + x_N^\top P x_N + R \sum_{k=0}^{N-1} u_k^2$$

**Kondensiertes QP.** Mit $\bar Q = \mathrm{blockdiag}(Q, \dots, Q, P)$ gilt $J = c + \tfrac12 u^\top H u + f^\top u$ mit

$$H = 2\left(\Gamma^\top \bar Q \Gamma + R\, I\right), \qquad f = F x_0, \qquad F = 2\, \Gamma^\top \bar Q\, \Phi$$

$$\min_{u}\ \tfrac12 u^\top H u + f^\top u \quad \text{unter} \quad -u_{\max} \le u_i \le u_{\max}, \quad i = 0, \dots, N-1$$

**Aktivmengen-Methode.** Je Durchlauf: Newton-Schritt auf den freien Stellen; trifft er eine Grenze, wird sie ins Arbeitsset genommen; ist der Schritt null, entscheidet das Vorzeichen des Gradienten $g = Hu + f$ an den festen Stellen
($g_i \ge 0$ unten, $g_i \le 0$ oben); die am stärksten verletzte Stelle wird freigegeben, sonst ist $u$ optimal.

**Beide Regeln.** Mit $u = -Kx$ aus der Riccati-Gleichung für den unendlichen Horizont:

$$u_{\text{Abschneiden}} = \mathrm{clip}\left(-Kx,\ -u_{\max},\ u_{\max}\right), \qquad u_{\text{MPC}} = u_0^\star(x)$$

Mit $P = P_\infty$ ist die erste Stelle der unbeschränkten Lösung für jeden Horizont $-Kx$; sobald die Grenze in der Folge greift, gilt das nicht mehr. Der Aufschlag eines Verfahrens ist
$100\,(\text{Kosten}/\text{Referenzkosten} - 1)$ in Prozent.

Die Formeln beschreiben das Modell der Demo vollständig bis auf Zustandsgrenzen und Terminalmengen, die nicht vorkommen. Quellen: Rawlings/Mayne/Diehl, *Model Predictive Control*, 2. Aufl., Kap. 2 (Abschn. 2.2 und das Skalar-Beispiel);
Borrelli/Bemporad/Morari, *Predictive Control for Linear and Hybrid Systems*, Kap. 11.3.1. Seitenangaben stehen in der README; sie sind am Original noch nicht geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
