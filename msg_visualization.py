"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import msg_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.22))
    return lock_axes(fig)


# ---------------------------------------------------------------------------------------------------------------- Experiment 1
def build_hand_contours(x: float, u_opt, umax: float = 1.0):
    """Höhenlinien der Kosten V(u₀, u₁) des Skalars (N = 2, Faktor ½) über der Ebene der Stellsignale, die Box |u| ≤ 1 und das Optimum."""
    lim = 2.4
    g = np.linspace(-lim, lim, 121)
    U0, U1 = np.meshgrid(g, g)
    V = 0.5 * (x ** 2 + U0 ** 2 + (x + U0) ** 2 + U1 ** 2 + (x + U0 + U1) ** 2)
    fig = go.Figure(go.Contour(x=g, y=g, z=V, ncontours=18, colorscale="Blues", reversescale=True, showscale=False, line=dict(width=0.6),
                               hovertemplate="u₀ = %{x:.2f}, u₁ = %{y:.2f}: V = %{z:.2f}<extra></extra>"))
    fig.add_shape(type="rect", x0=-umax, x1=umax, y0=-umax, y1=umax, line=dict(color=C.COLOR_CLIP, width=2), fillcolor="rgba(192,80,77,0.07)")
    unc = (-0.6 * x, -0.2 * x)
    fig.add_trace(go.Scatter(x=[unc[0]], y=[unc[1]], mode="markers", marker=dict(symbol="x", size=11, color="black"), name="ohne Grenze"))
    fig.add_trace(go.Scatter(x=[u_opt[0]], y=[u_opt[1]], mode="markers", marker=dict(symbol="circle", size=12, color=C.COLOR_MPC, line=dict(color="white", width=1)), name="QP mit Grenze"))
    fig.update_xaxes(title_text="u₀", range=[-lim, lim])
    fig.update_yaxes(title_text="u₁", range=[-lim, lim])
    return _base(fig, height=380)


def build_hand_cost(cum, predicted: float, driven: float):
    """Summe der gefahrenen Stufenkosten über den Takten, dazu die Voraussage V_N⁰ (gestrichelt) und der Endwert."""
    ks = list(range(len(cum)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=cum, mode="lines+markers", line=dict(color=C.COLOR_NEUTRAL, width=3), name="gefahrene Summe"))
    fig.add_hline(y=predicted, line=dict(color=C.COLOR_MPC, dash="dash", width=2), annotation_text=f"Voraussage V = {predicted:.2f}".replace(".", ","), annotation_position="top left")
    fig.add_hline(y=driven, line=dict(color=C.COLOR_CLIP, dash="dot", width=2), annotation_text=f"gefahren (unendlich) = {driven:.2f}".replace(".", ","), annotation_position="bottom right")
    fig.update_xaxes(title_text="Takt k")
    fig.update_yaxes(title_text="Kosten (Faktor ½)")
    return _base(fig, height=380)


# ---------------------------------------------------------------------------------------------------------------- Experiment 2
def build_skalar_law(r: dict):
    """Erstes Stellsignal des QP und des abgeschnittenen unbeschränkten Gesetzes über dem Zustand (jeder 10. Punkt)."""
    xs, m, c = r["xs"][::10], r["mpc"][::10], r["clip"][::10]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=c, mode="lines", line=dict(color=C.COLOR_CLIP, width=6), name="abgeschnittenes unbeschränktes Gesetz"))
    fig.add_trace(go.Scatter(x=xs, y=m, mode="lines", line=dict(color=C.COLOR_MPC, width=2, dash="dot"), name="QP (MPC)"))
    fig.update_xaxes(title_text="Zustand x")
    fig.update_yaxes(title_text="erstes Stellsignal u₀")
    return _base(fig)


def _heat(pts, values, nx, ny, x1, x2):
    """Werte je Startzustand (Ursprung fehlt) als Matrix x₂ (Zeilen) × x₁ (Spalten)."""
    z = np.full((ny, nx), np.nan)
    xs1 = list(np.round(x1, 10))
    xs2 = list(np.round(x2, 10))
    for p, v in zip(pts, values):
        z[xs2.index(round(float(p[1]), 10)), xs1.index(round(float(p[0]), 10))] = v
    return z


def build_wagen_map(pts, mehrkosten, x1_axis, x2_axis):
    """Karte der Mehrkosten von „LQR, dann abschneiden“ gegen MPC in % über den Startzuständen (x₁ Ort, x₂ Geschwindigkeit)."""
    z = _heat(pts, mehrkosten, len(x1_axis), len(x2_axis), x1_axis, x2_axis)
    fig = go.Figure(go.Heatmap(x=x1_axis, y=x2_axis, z=z, colorscale=[[0, "#f4f4f4"], [0.15, "#fdd49e"], [1, "#b30000"]], zmin=0,
                               colorbar=dict(title=dict(text="Mehrkosten %", side="top"), nticks=4), hovertemplate="x₁ = %{x}, x₂ = %{y}: %{z:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Ort x₁")
    fig.update_yaxes(title_text="Geschwindigkeit x₂")
    return _base(fig, height=380, legend=False)


# ---------------------------------------------------------------------------------------------------------------- Experiment 3
def build_scaling(sc: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc["faktoren"], y=sc["clip"], mode="lines+markers", name="Abschneiden", line=dict(color=C.COLOR_CLIP, width=3)))
    fig.add_trace(go.Scatter(x=sc["faktoren"], y=sc["mpc"], mode="lines+markers", name="MPC", line=dict(color=C.COLOR_MPC, width=2, dash="dot")))
    fig.update_xaxes(title_text="Faktor c", type="log")
    fig.update_yaxes(title_text="Kosten(c·x₀, c·u_max) / (c² · Kosten(x₀, u_max))", range=[0.99, 1.01])
    return _base(fig, height=320)


def build_distance(ab: dict):
    fig = go.Figure(go.Scatter(x=ab["dist"], y=ab["mehrkosten"], mode="markers", marker=dict(color=C.COLOR_CLIP, size=6, opacity=0.7),
                               hovertemplate="‖x₀‖/u_max = %{x:.1f}: %{y:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Abstand ‖x₀‖ / u_max")
    fig.update_yaxes(title_text="Mehrkosten des Abschneidens (%)", rangemode="tozero")
    return _base(fig, height=320, legend=False)


# ---------------------------------------------------------------------------------------------------------------- Experiment 4
def build_horizon(szenarien: dict, selected_key: str | None, n_sel: int, terminal: str):
    """Maximaler Aufschlag von MPC gegen die Referenz (vorgerechnet) über dem Horizont, je Szenario; Aufschläge unter 0,001 % liegen auf der Linie 0,001 %."""
    fig = go.Figure()
    farben = {"1|1": C.COLOR_MPC, "1|0.5": C.COLOR_CLIP, "0.1|1": C.COLOR_NEUTRAL}
    namen = {"1|1": "u_max = 1, R = 1", "1|0.5": "u_max = 0,5, R = 1", "0.1|1": "u_max = 1, R = 0,1"}
    for key, sc in szenarien.items():
        ns = sorted(int(n) for n in sc["mpc"])
        ys = [max(sc["mpc"][str(n)]["max"], 1e-3) for n in ns]
        fig.add_trace(go.Scatter(x=ns, y=ys, mode="lines+markers", name=namen.get(key, key), line=dict(color=farben.get(key), width=3 if key == selected_key else 1.5)))
    fig.add_vline(x=n_sel, line=dict(color="grey", dash="dash"), annotation_text=f"N = {n_sel}", annotation_position="top")
    fig.update_xaxes(title_text="Horizont N", type="log", tickvals=[1, 2, 3, 5, 10, 20], ticktext=["1", "2", "3", "5", "10", "20"])
    fig.update_yaxes(title_text="größter Aufschlag gegen die Referenz (%)", type="log")
    return _base(fig, height=340)


# ---------------------------------------------------------------------------------------------------------------- Experiment 5
def build_sequence(tr: dict):
    """Optimale Stellfolge des QP (Balken) mit der unbeschränkten Lösung (Punkte) und den Grenzen ±u_max."""
    u = tr["result"].u
    ks = list(range(len(u)))
    labels = [f"u{k}" for k in ks]
    farben = [C.COLOR_CLIP if abs(abs(v) - tr["umax"]) <= 1e-9 * tr["umax"] else C.COLOR_MPC for v in u]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=list(u), marker_color=farben, name="QP mit Grenze (rot: an der Grenze)"))
    fig.add_trace(go.Scatter(x=labels, y=list(tr["unbeschraenkt"]), mode="markers", marker=dict(symbol="x", size=9, color="black"), name="ohne Grenze"))
    fig.add_hline(y=tr["umax"], line=dict(color="grey", dash="dash"))
    fig.add_hline(y=-tr["umax"], line=dict(color="grey", dash="dash"))
    fig.update_yaxes(title_text="Stellsignal")
    return _base(fig, height=320)


# ---------------------------------------------------------------------------------------------------------------- Experiment 6
def build_stability_map(m: dict, which: str, title: str):
    """Karte des Gitters [−6, 6]²: grün = konvergiert, rot = nicht; die Umrisslinie ist die per linearem Programm steuerbare Menge."""
    n = m["n"]
    axis = np.linspace(C.UNSTABLE_GRID[0], C.UNSTABLE_GRID[1], n)
    full = np.zeros((n, n))
    ctrl = np.zeros((n, n))
    pts = m["pts"]
    for i, p in enumerate(pts):
        ix = int(round((p[0] - C.UNSTABLE_GRID[0]) / C.UNSTABLE_GRID[2]))
        iy = int(round((p[1] - C.UNSTABLE_GRID[0]) / C.UNSTABLE_GRID[2]))
        full[iy, ix] = 1.0 if m[which][i] else 0.0
        ctrl[iy, ix] = 1.0 if m["steuerbar"][i] else 0.0
    mid = n // 2
    full[mid, mid] = 1.0
    ctrl[mid, mid] = 1.0
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=axis, y=axis, z=full, zmin=0, zmax=1, colorscale=[[0, C.COLOR_BAD], [1, C.COLOR_OK]], showscale=False, xgap=1, ygap=1,
                             hovertemplate="x₁ = %{x}, x₂ = %{y}: %{z}<extra></extra>"))
    fig.add_trace(go.Contour(x=axis, y=axis, z=ctrl, showscale=False, contours=dict(start=0.5, end=0.5, size=1, coloring="lines"), colorscale=[[0, "black"], [1, "black"]], line=dict(width=2), hoverinfo="skip",
                             name="steuerbar"))
    fig.update_xaxes(title_text="Ort x₁")
    fig.update_yaxes(title_text="Geschwindigkeit x₂", scaleanchor="x", scaleratio=1)
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return _base(fig, height=360, legend=False)


# ---------------------------------------------------------------------------------------------------------------- Primäransicht
def build_trajectory(tr: dict):
    """Regelkreis aus einem Startzustand: Ort x₁ und Stellsignal u über den Takten, MPC gegen „LQR, dann abschneiden“."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Ort x₁", "Stellsignal u"))
    for name, lp, color, dash, width in (("LQR, dann abschneiden", tr["clip"], C.COLOR_CLIP, "solid", 4), ("MPC", tr["mpc"], C.COLOR_MPC, "dot", 2.5)):
        k = list(range(len(lp.us)))
        fig.add_trace(go.Scatter(x=list(range(len(lp.xs))), y=list(lp.xs[:, 0]), mode="lines", line=dict(color=color, dash=dash, width=width), name=name, legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=k, y=list(lp.us), mode="lines", line=dict(color=color, dash=dash, width=width), name=name, legendgroup=name, showlegend=False,
                                 line_shape="hv"), row=2, col=1)
    for sign in (1, -1):
        fig.add_hline(y=sign * tr["umax"], line=dict(color="grey", dash="dash", width=1), row=2, col=1)
    fig.update_xaxes(title_text="Takt k", row=2, col=1)
    return _base(fig, height=430)
