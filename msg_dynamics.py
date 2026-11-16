"""Regelkreis: Strecken, Riccati-Gewinn, die beiden Regeln „LQR, dann abschneiden“ und MPC mit Stellgrenze, gefahrene Kosten, Gitter von Startzuständen.

Strecken: `wagen` x⁺ = [[a, 1], [0, 1]] x + (0; 1) u (a = 1: Doppelintegrator, Ts = 1; a > 1: offen instabil), `skalar` x⁺ = x + u. Q = I.
Kosten eines Laufs: Σ_k (x_k'Q x_k + R u_k²) bis |x| < 1e-9 (höchstens STEPS_MAX Schritte), ohne Faktor ½.

„Abschneiden“ heißt in der App zwei getrennt benannte Dinge:
  * **LQR-Gesetz abschneiden**: u = clip(−K x) mit dem unbeschränkten Riccati-Gewinn K (Kosten der Vergleiche);
  * **unbeschränkte N-Schritt-Folge abschneiden**: das erste Element der unbeschränkten Lösung des N-Schritt-Problems, abgeschnitten (nur das erste Stellsignal, Vergleich mit dem QP)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.linalg as la

import msg_constants as C
import msg_qp as QP


# ---------------------------------------------------------------------------------------------------------------- Strecken, Riccati
def plant(kind: str, a: float = 1.0):
    """(A, B, Q) der Strecke."""
    if kind == "skalar":
        return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    if kind == "wagen":
        return np.array([[a, 1.0], [0.0, 1.0]]), np.array([[0.0], [1.0]]), np.eye(2)
    raise ValueError(f"unbekannte Strecke {kind!r}")


def dare(A, B, Q, R):
    """(P, K) der diskreten Riccati-Gleichung; u = −K x ist das unbeschränkte LQR-Gesetz für den unendlichen Horizont."""
    Rm = np.array([[float(R)]])
    P = la.solve_discrete_are(A, B, Q, Rm)
    K = np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A)
    return P, K


# ---------------------------------------------------------------------------------------------------------------- MPC-Problem und Regeln
@dataclass(frozen=True, eq=False)
class MPCProblem:
    kind: str
    a: float
    N: int
    R: float
    umax: float
    terminal: str
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    P: np.ndarray
    K: np.ndarray        # Riccati-Gewinn (unbeschränkt, unendlicher Horizont)
    H: np.ndarray
    F: np.ndarray
    Cmat: np.ndarray
    lo: np.ndarray
    hi: np.ndarray



@lru_cache(maxsize=128)
def build_mpc(kind: str, a: float, N: int, R: float, umax: float, terminal: str) -> MPCProblem:
    A, B, Q = plant(kind, a)
    P_dare, K = dare(A, B, Q, R)
    P = P_dare if terminal == "DARE" else Q
    _, _, H, F, Cm = QP.condensed(N, R, P, A, B, Q)
    return MPCProblem(kind, a, N, R, umax, terminal, A, B, Q, P, K, H, F, Cm, -umax * np.ones(N), umax * np.ones(N))


def unconstrained_first_gain(prob: MPCProblem) -> np.ndarray:
    """Erstes Stellsignal der unbeschränkten N-Schritt-Lösung als Zeile: u₀ = k x (mit P = DARE gleich −K für jedes N)."""
    return -np.linalg.solve(prob.H, prob.F)[0]


def clip_value(v, umax):
    return float(np.clip(v, -umax, umax))


class ClipLaw:
    """LQR, dann abschneiden: u = clip(−K x, −u_max, u_max)."""

    def __init__(self, K, umax):
        self.K = np.asarray(K, float)
        self.umax = float(umax)

    def __call__(self, x):
        return clip_value(-(self.K @ x)[0], self.umax)


class MPCLaw:
    """MPC mit Stellgrenze: löst in jedem Takt das QP (Kaltstart) und wendet das erste Element an. Zählt Aufrufe, Iterationen, Muster und Rückfälle."""

    def __init__(self, prob: MPCProblem, record=False):
        self.prob = prob
        self.record = record
        self.calls = 0
        self.fallbacks = 0
        self.iterations: list[int] = []
        self.active: list[int] = []
        self.patterns: set = set()

    def solve(self, x, trace=False) -> QP.QPResult:
        p = self.prob
        return QP.qp_solve(p.H, p.F @ x, p.lo, p.hi, trace=trace)

    def __call__(self, x):
        r = self.solve(x)
        if self.record:
            self.calls += 1
            if r.fallback:
                self.fallbacks += 1
            else:
                self.iterations.append(r.iterations)
            pat = QP.sign_pattern(r.u, self.prob.lo, self.prob.hi)
            self.active.append(int(sum(abs(v) for v in pat)))
            self.patterns.add(pat)
        return float(r.u[0])


# ---------------------------------------------------------------------------------------------------------------- Regelkreis
@dataclass
class Loop:
    cost: float
    end_norm: float
    max_norm: float
    steps: int
    xs: np.ndarray | None = None      # Zustände x_0 … x_T (nur bei record)
    us: np.ndarray | None = None


def run_loop(law, x0, A, B, Q, R, T=C.STEPS_MAX, tol=C.CONV_TOL, record=False, blow=1e12) -> Loop:
    """Fährt den Regelkreis bis |x| < tol oder T Schritte und summiert Σ (x'Qx + R u²)."""
    x = np.array(x0, float)
    b = B[:, 0]
    cost = 0.0
    max_norm = float(np.linalg.norm(x))
    xs, us = ([x.copy()], []) if record else (None, None)
    steps = 0
    for _ in range(T):
        u = law(x)
        cost += float(x @ Q @ x) + R * u * u
        x = A @ x + b * u
        steps += 1
        nrm = float(np.linalg.norm(x))
        max_norm = max(max_norm, nrm)
        if record:
            xs.append(x.copy())
            us.append(u)
        if nrm < tol or nrm > blow:
            break
    return Loop(cost, float(np.linalg.norm(x)), max_norm, steps, None if xs is None else np.array(xs), None if us is None else np.array(us))


def run_clip_batch(pts, K, umax, A, B, Q, R, T=C.STEPS_MAX, tol=C.CONV_TOL):
    """Dasselbe für das abgeschnittene LQR-Gesetz, vektorisiert über viele Startzustände (m × n). Gibt (Kosten, Endnorm, größte Norm, Schritte)."""
    X = np.array(pts, float)
    m = len(X)
    cost = np.zeros(m)
    max_norm = np.linalg.norm(X, axis=1)
    steps = np.zeros(m, int)
    active = np.ones(m, bool)
    b = B[:, 0]
    Kv = np.asarray(K, float).reshape(-1)
    for _ in range(T):
        if not active.any():
            break
        u = np.clip(-(X @ Kv), -umax, umax)
        cost += np.where(active, np.einsum("ij,jk,ik->i", X, Q, X) + R * u * u, 0.0)
        Xn = X @ A.T + np.outer(u, b)
        X = np.where(active[:, None], Xn, X)
        steps += active
        nrm = np.linalg.norm(X, axis=1)
        max_norm = np.where(active, np.maximum(max_norm, nrm), max_norm)
        active = active & (nrm >= tol)
    return cost, np.linalg.norm(X, axis=1), max_norm, steps


# ---------------------------------------------------------------------------------------------------------------- Gitter
def grid_axis(spec):
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + step / 2, step), 10)


def grid_points(spec1, spec2, drop_origin=True):
    pts = [(float(a), float(b)) for a in grid_axis(spec1) for b in grid_axis(spec2)]
    return [p for p in pts if not (drop_origin and p == (0.0, 0.0))]


@dataclass
class GridRun:
    pts: np.ndarray
    cost_clip: np.ndarray
    end_clip: np.ndarray
    cost_mpc: np.ndarray
    end_mpc: np.ndarray
    iterations_mean: float
    fallbacks: int


def run_grid(prob: MPCProblem, pts, T=C.STEPS_MAX) -> GridRun:
    """Gefahrene Kosten von „LQR, dann abschneiden“ und MPC (gleiche Strecke, gleiche Grenze) an allen Startzuständen."""
    P = np.array(pts, float)
    cc, ec, _, _ = run_clip_batch(P, prob.K, prob.umax, prob.A, prob.B, prob.Q, prob.R, T)
    law = MPCLaw(prob, record=True)
    cm = np.zeros(len(P))
    em = np.zeros(len(P))
    for i, x0 in enumerate(P):
        lp = run_loop(law, x0, prob.A, prob.B, prob.Q, prob.R, T)
        cm[i], em[i] = lp.cost, lp.end_norm
    its = np.array(law.iterations) if law.iterations else np.array([0])
    return GridRun(P, cc, ec, cm, em, float(its.mean()), law.fallbacks)


def reference_costs(R, umax, pts, T=C.STEPS_MAX, kind="wagen", a=1.0, N=C.REF_N):
    """Kosten der Referenz (MPC mit N = 60, Endkosten DARE) an den Startzuständen; schwer, deshalb in der App vorgerechnet."""
    prob = build_mpc(kind, a, N, R, umax, "DARE")
    law = MPCLaw(prob)
    return np.array([run_loop(law, x0, prob.A, prob.B, prob.Q, R, T).cost for x0 in pts])


def compare_counts(c_clip, c_mpc, tol=C.EQUAL_REL):
    """Zählung MPC gegen Abschneiden: (billiger, gleich, teurer, größter Vorteil in %, größter Nachteil in %). Gleich: relativ höchstens tol."""
    c_clip = np.asarray(c_clip, float)
    c_mpc = np.asarray(c_mpc, float)
    rel = c_mpc / c_clip - 1.0
    cheaper = int(np.sum(rel < -tol))
    dearer = int(np.sum(rel > tol))
    equal = len(rel) - cheaper - dearer
    adv = float(100 * np.max(1 - c_mpc / c_clip)) if len(rel) else 0.0
    dis = float(100 * np.max(rel)) if len(rel) else 0.0
    return cheaper, equal, dearer, max(adv, 0.0), max(dis, 0.0)


# ---------------------------------------------------------------------------------------------------------------- Konvergenz auf dem Gitter (Experiment 6)
def converges(law, x0, A, B, T=C.STEPS_MAX, tol=C.CONV_COUNT_TOL, blow=1e6):
    """Norm unter tol nach höchstens T Schritten, ohne dass sie blow übersteigt."""
    x = np.array(x0, float)
    b = B[:, 0]
    for _ in range(T):
        x = A @ x + b * law(x)
        nrm = float(np.linalg.norm(x))
        if nrm < tol:
            return True
        if nrm > blow:
            return False
    return False


def controllable_mask(A, B, pts, umax=1.0, steps=C.CTRL_STEPS):
    """Welche Startzustände lassen sich mit |u| ≤ umax in genau `steps` Schritten auf den Ursprung steuern? Lineares Programm (Zulässigkeit)."""
    from scipy.optimize import linprog

    b = B[:, 0]
    Gam = np.column_stack([np.linalg.matrix_power(A, steps - 1 - j) @ b for j in range(steps)])
    AN = np.linalg.matrix_power(A, steps)
    out = []
    for p in pts:
        r = linprog(np.zeros(steps), A_eq=Gam, b_eq=-AN @ np.array(p, float), bounds=[(-umax, umax)] * steps, method="highs")
        out.append(r.status == 0)
    return np.array(out)
