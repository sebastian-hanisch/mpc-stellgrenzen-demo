"""Das quadratische Programm (QP) des MPC mit Stellgrenze: kondensierte Matrizen und eine Aktivmengen-Methode für Boxgrenzen (numpy, SciPy nur im Rückfall).

Konvention (ohne Faktor ½ in den Stufenkosten, wie Borrelli/Bemporad/Morari und Åström/Murray):
    J(x0, u) = x0'Q x0 + Σ_{k=1}^{N-1} x_k'Q x_k + x_N'P x_N + R Σ u_k²,   x_{k+1} = A x_k + B u_k.
Kondensiert (Stellfolge u = (u_0 … u_{N-1}) als Unbekannte): J = c + ½ u'H u + f'u mit
    H = 2 (G'Q̄G + R·I),  f = F x0,  F = 2 G'Q̄Φ,  c = x0'(Q + Φ'Q̄Φ) x0;   Q̄ = blockdiag(Q, …, Q, P).
Gleichungen in eigener Herleitung; der Aufbau entspricht dem Standardlehrbuchverfahren (README, Quellen).

Die Aktivmengen-Methode ist ein Lehrbuchverfahren (primal, nur Boxgrenzen), hier neu geschrieben und nicht aus einer gelesenen Quelle zitiert."""

from dataclasses import dataclass, field

import numpy as np

STEP_TOL = 1e-9        # Schritt gilt als null, wenn er relativ zum Boxradius kleiner ist
MULT_TOL = 1e-9        # Multiplikator gilt als verletzt, wenn er relativ zur Gradientengröße größer ist
MAXIT = 500            # Iterationsgrenze der Aktivmengen-Methode, danach Rückfall


class QPNoConvergence(RuntimeError):
    """Die Aktivmengen-Methode ist nach MAXIT Iterationen nicht zum Ende gekommen (schlecht konditioniertes H)."""


@dataclass
class QPResult:
    u: np.ndarray
    iterations: int                 # Durchläufe der Hauptschleife einschließlich des letzten, der die Optimalität feststellt; -1 bei Rückfall
    fallback: bool = False          # True: die Lösung stammt aus dem Rückfall (bvls), nicht aus der Aktivmengen-Methode
    trace: list = field(default_factory=list)


def condensed(N, R, P, A, B, Q):
    """Gibt (Phi, G, H, F, C) zurück: Phi (nN × n), G (nN × N), H (N × N), F (N × n), C (n × n). Ein Eingang (B ist n × 1)."""
    n = A.shape[0]
    b = B[:, 0]
    powers = [np.eye(n)]
    for _ in range(N):
        powers.append(A @ powers[-1])
    Phi = np.vstack(powers[1:N + 1])
    G = np.zeros((n * N, N))
    for k in range(1, N + 1):
        for j in range(k):
            G[n * (k - 1):n * k, j] = powers[k - 1 - j] @ b
    Qbar = np.zeros((n * N, n * N))
    for k in range(1, N):
        Qbar[n * (k - 1):n * k, n * (k - 1):n * k] = Q
    Qbar[n * (N - 1):, n * (N - 1):] = P
    H = 2.0 * (G.T @ Qbar @ G + R * np.eye(N))
    F = 2.0 * G.T @ Qbar @ Phi
    C = Q + Phi.T @ Qbar @ Phi
    return Phi, G, H, F, C


def qp_value(u, H, f, c=0.0):
    return float(0.5 * u @ H @ u + f @ u + c)


def qp_active_set(H, f, lo, hi, trace=False, maxit=MAXIT) -> QPResult:
    """min ½u'Hu + f'u unter lo ≤ u ≤ hi (H positiv definit). Start bei u = clip(0), Kaltstart ohne Vorwissen.

    Je Durchlauf: Newton-Schritt auf den freien Stellen (die Stellen an einer Grenze bleiben fest). Trifft er eine Grenze, wird sie ins Arbeitsset
    genommen; ist der Schritt null, entscheidet das Vorzeichen des Gradienten an den festen Stellen (unten fest braucht g ≥ 0, oben fest g ≤ 0):
    die am stärksten verletzte Stelle wird freigegeben, sonst ist u optimal. Toleranzen sind relativ (zum Boxradius und zur Gradientengröße)."""
    f = np.asarray(f, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    N = len(f)
    u = np.clip(np.zeros(N), lo, hi)
    scale = float(np.max(hi - lo)) / 2.0 if N else 0.0
    if scale <= 0.0:                                   # lo = hi: nichts frei
        return QPResult(u, 1, False, [])
    at_lo = u <= lo
    at_hi = (u >= hi) & ~at_lo
    gscale = max(float(np.max(np.abs(f))), float(np.max(np.abs(H))) * scale, 1e-300)
    steps = []
    for it in range(1, maxit + 1):
        free = ~(at_lo | at_hi)
        g = H @ u + f
        p = np.zeros(N)
        if free.any():
            p[free] = -np.linalg.solve(H[np.ix_(free, free)], g[free])
        if trace:
            steps.append({"iteration": it, "u": u.copy(), "unten": at_lo.copy(), "oben": at_hi.copy(), "wert": qp_value(u, H, f)})
        if float(np.max(np.abs(p))) <= STEP_TOL * scale:
            viol = np.where(at_lo, -g, 0.0) + np.where(at_hi, g, 0.0)      # > 0: Multiplikator hat das falsche Vorzeichen
            i = int(np.argmax(viol))
            if viol[i] <= MULT_TOL * gscale:
                if trace:
                    steps[-1]["aktion"] = "optimal"
                return QPResult(u, it, False, steps)
            at_lo[i] = False
            at_hi[i] = False
            if trace:
                steps[-1]["aktion"] = f"Grenze bei u{i} freigegeben"
            continue
        alpha, block, side = 1.0, -1, 0
        for i in np.nonzero(free)[0]:
            if p[i] > 0.0:
                a = max((hi[i] - u[i]) / p[i], 0.0)
                if a < alpha:
                    alpha, block, side = a, int(i), 1
            elif p[i] < 0.0:
                a = max((lo[i] - u[i]) / p[i], 0.0)
                if a < alpha:
                    alpha, block, side = a, int(i), -1
        u = u + alpha * p
        if block >= 0:
            if side > 0:
                u[block] = hi[block]
                at_hi[block] = True
            else:
                u[block] = lo[block]
                at_lo[block] = True
            if trace:
                steps[-1]["aktion"] = f"Schritt bis zur {'oberen' if side > 0 else 'unteren'} Grenze von u{block}"
        elif trace:
            steps[-1]["aktion"] = "voller Schritt (Newton)"
    raise QPNoConvergence(f"keine Lösung nach {maxit} Iterationen")


def qp_bvls(H, f, lo, hi):
    """Rückfall: beschränkte kleinste Quadrate (Cholesky-Umformung) mit großer Iterationsgrenze; die Voreinstellung der Iterationszahl ist bei N ≥ 20 zu ungenau."""
    from scipy.optimize import lsq_linear

    L = np.linalg.cholesky(H)
    target = -np.linalg.solve(L, f)
    return lsq_linear(L.T, target, bounds=(lo, hi), method="bvls", tol=1e-14, max_iter=100000).x


def qp_solve(H, f, lo, hi, trace=False) -> QPResult:
    """Aktivmengen-Methode; bricht sie bei schlecht konditioniertem H ab, gilt der Rückfall (`fallback=True`, `iterations=-1`)."""
    try:
        return qp_active_set(H, f, lo, hi, trace=trace)
    except (QPNoConvergence, np.linalg.LinAlgError):
        return QPResult(qp_bvls(H, f, np.asarray(lo, float), np.asarray(hi, float)), -1, True, [])


def sign_pattern(u, lo, hi, tol=1e-9):
    """Vorzeichenmuster der Stellfolge: −1 an der unteren, +1 an der oberen Grenze, 0 dazwischen (Muster in den Bahnen, nicht Regionen eines expliziten Gesetzes)."""
    u = np.asarray(u, float)
    scale = max(float(np.max(np.asarray(hi) - np.asarray(lo))), 1e-300)
    return tuple(int(-1) if abs(v - l) <= tol * scale else int(1) if abs(v - h) <= tol * scale else 0 for v, l, h in zip(u, lo, hi))
