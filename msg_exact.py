"""Exakte Handrechnung mit Brüchen (`fractions`) für Experiment 1 und das Wagen-Handbeispiel: Box-QP durch Aufzählung aller Muster (unten / oben / frei je Stelle).

Skalar (Rawlings Bsp. 2.5, Kapitel 2, hier in eigener Rechnung): x⁺ = x + u, Stufenkosten ½(x² + u²), Endkosten ½x², N = 2, |u| ≤ 1. Mit Faktor ½.
Wagen-Handbeispiel: Doppelintegrator Ts = 1, Q = P = I, R = 1, N = 2, |u| ≤ ½, x0 = (7; −2), Kosten ohne Faktor ½ (Endkosten P = I, damit alles rational bleibt).
Aufzählung statt Aktivmengen-Methode: ein zweiter Rechenweg gegenüber `msg_qp.py` (die Tests prüfen beide gegeneinander)."""

import itertools
from fractions import Fraction as Fr


def solve_frac(M, b):
    """Gauß-Elimination mit Brüchen (M klein, regulär)."""
    n = len(b)
    a = [list(M[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[p] = a[p], a[c]
        for r in range(n):
            if r != c and a[r][c] != 0:
                fac = a[r][c] / a[c][c]
                a[r] = [x - fac * y for x, y in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def box_qp_enum(H, f, lo, hi):
    """min ½u'Hu + f'u unter lo ≤ u ≤ hi exakt: alle 3^N Muster, je Muster der stationäre Punkt der freien Stellen, der kleinste zulässige Wert gewinnt.
    Gibt (u, Wert ohne Konstante) zurück."""
    N = len(f)
    lo, hi = Fr(lo), Fr(hi)
    best, bestv = None, None
    for pat in itertools.product((-1, 0, 1), repeat=N):
        u = [lo if p < 0 else hi if p > 0 else None for p in pat]
        fr = [i for i in range(N) if pat[i] == 0]
        if fr:
            Mff = [[H[i][j] for j in fr] for i in fr]
            rhs = [-(f[i] + sum(H[i][j] * u[j] for j in range(N) if u[j] is not None)) for i in fr]
            for i, v in zip(fr, solve_frac(Mff, rhs)):
                u[i] = v
        if all(lo <= v <= hi for v in u):
            v = Fr(1, 2) * sum(u[i] * H[i][j] * u[j] for i in range(N) for j in range(N)) + sum(f[i] * u[i] for i in range(N))
            if bestv is None or v < bestv:
                best, bestv = u, v
    return best, bestv


def sat(v, lim=Fr(1)):
    return max(-lim, min(lim, Fr(v)))


# ---------------------------------------------------------------------------------------------------------------- Skalar, N = 2
SCALAR_H = [[Fr(3), Fr(1)], [Fr(1), Fr(2)]]          # aus V = ½[x² + u0² + (x + u0)² + u1² + (x + u0 + u1)²] = ½u'Hu + (2x; x)'u + (3/2)x²


def scalar_n2(x):
    """QP des Skalars bei N = 2 für den Startwert x (Bruch oder Zahl): (u0, u1), vorhergesagter Wert V_N⁰(x) (mit Faktor ½) und welche Stellen an der Grenze liegen."""
    x = Fr(x)
    u, v = box_qp_enum(SCALAR_H, [2 * x, x], Fr(-1), Fr(1))
    at = [("unten" if ui == -1 else "oben" if ui == 1 else "frei") for ui in u]
    return u, v + Fr(3, 2) * x * x, at


def scalar_loop(x0):
    """Regelkreis x⁺ = x − sat(3x/5) ab x0 (Gesetz u₀ = −sat(3x/5)) exakt: gefahrene Kosten bis ins Unendliche (mit Faktor ½) und die Folge der Zustände bis |x| ≤ 5/3.
    Ab |x| ≤ 5/3 gilt x⁺ = (2/5)x mit Stufenkosten (17/25)x² und der Summe (17/25)x²/(1 − 4/25) = (17/21)x²."""
    x = Fr(x0)
    cost = Fr(0)
    folge = [x]
    while abs(x) > Fr(5, 3):
        u = -sat(Fr(3, 5) * x)
        cost += Fr(1, 2) * (x * x + u * u)
        x += u
        folge.append(x)
    return cost + Fr(17, 21) * x * x, folge


# ---------------------------------------------------------------------------------------------------------------- Wagen, N = 2
def _matmul(M, N):
    return [[sum(M[i][k] * N[k][j] for k in range(len(N))) for j in range(len(N[0]))] for i in range(len(M))]


def wagen_hand(x0=(7, -2), umax=Fr(1, 2)):
    """Handbeispiel Wagen, N = 2, Q = P = I, R = 1: H, f, unbeschränkte Lösung, MPC-Lösung, abgeschnittene unbeschränkte Folge und die drei Kosten (Brüche)."""
    x0 = [Fr(v) for v in x0]
    umax = Fr(umax)
    # x1 = (a + b, b + u0), x2 = (a + 2b + u0, b + u0 + u1); Kosten |x0|² + |x1|² + R u0² + R u1² + |x2|² (Endkosten P = I)
    Gam = [[0, 0], [1, 0], [1, 0], [1, 1]]               # (x1; x2) = Phi x0 + Gam u
    Phi = [[1, 1], [0, 1], [1, 2], [0, 1]]
    Gam = [[Fr(v) for v in r] for r in Gam]
    Phi = [[Fr(v) for v in r] for r in Phi]
    GtG = _matmul([list(r) for r in zip(*Gam)], Gam)
    H = [[2 * GtG[i][j] + (2 if i == j else 0) for j in range(2)] for i in range(2)]
    GtPhi = _matmul([list(r) for r in zip(*Gam)], Phi)
    f = [2 * sum(GtPhi[i][j] * x0[j] for j in range(2)) for i in range(2)]
    unc = solve_frac(H, [-v for v in f])
    mpc, _ = box_qp_enum(H, f, -umax, umax)
    clipped = [sat(v, umax) for v in unc]

    def J(u):
        x = list(x0)
        tot = Fr(0)
        for k in range(2):
            tot += x[0] ** 2 + x[1] ** 2 + u[k] ** 2
            x = [x[0] + x[1], x[1] + u[k]]
        return tot + x[0] ** 2 + x[1] ** 2

    g = [sum(H[i][j] * mpc[j] for j in range(2)) + f[i] for i in range(2)]
    return {"H": H, "f": f, "unbeschraenkt": unc, "mpc": mpc, "abgeschnitten": clipped, "J_unbeschraenkt": J(unc), "J_mpc": J(mpc), "J_abgeschnitten": J(clipped),
            "gradient": g}
