"""Gemeinsame Hilfen: das Handbeispiel (Wagen, N = 2) und ein exakter Rechner für die Stellfolgen-Kosten, unabhängig vom Kern geschrieben (DEMO-PLAYBOOK §1b)."""

import itertools
import sys
from fractions import Fraction as Fr
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cost_direct(u, x0, R, P, A=((1, 1), (0, 1)), B=(0, 1)):
    """Kosten J = x0'x0 + Σ_{k=1}^{N-1} x_k'x_k + x_N'P x_N + R Σ u² durch Hinschreiben der Rekursion (Brüche oder Zahlen), Q = I, ein Eingang."""
    x = list(x0)
    tot = 0
    N = len(u)
    for k in range(N):
        tot += sum(v * v for v in x) + R * u[k] ** 2          # x_k'x_k (Q = I, k = 0 … N−1) und R u_k²
        x = [sum(A[i][j] * x[j] for j in range(len(x))) + B[i] * u[k] for i in range(len(x))]
    # Endzustand x_N mit P gewichtet (x_0 und die Zwischenzustände x_1 ... x_{N-1} stehen schon in der Schleife, x_N noch nicht)
    n = len(x)
    tot += sum(x[i] * P[i][j] * x[j] for i in range(n) for j in range(n))
    return tot


def hessian_by_differences(N, x0, R, P, **kw):
    """H und f des kondensierten QP aus der direkt summierten Kostenfunktion J (quadratisch): H_ii = J(e_i) + J(−e_i) − 2J(0), H_ij = J(e_i + e_j) − J(e_i) − J(e_j) + J(0),
    f_i = (J(e_i) − J(−e_i))/2. Ein vom Kern völlig unabhängiger Weg (nur die Kostendefinition, keine Matrixformel)."""
    def J(u):
        return cost_direct(u, x0, R, P, **kw)

    zero = [Fr(0)] * N
    e = lambda i, s=1: [Fr(s) if k == i else Fr(0) for k in range(N)]  # noqa: E731
    J0 = J(zero)
    H = [[Fr(0)] * N for _ in range(N)]
    f = [Fr(0)] * N
    for i in range(N):
        Jp, Jm = J(e(i)), J(e(i, -1))
        H[i][i] = Jp + Jm - 2 * J0
        f[i] = (Jp - Jm) / 2
        for j in range(i + 1, N):
            eij = [a + b for a, b in zip(e(i), e(j))]
            H[i][j] = H[j][i] = J(eij) - Jp - J(e(j)) + J0
    return H, f, J0


def enum_box_qp(H, f, lo, hi):
    """Alle 3^N Muster (unten, oben, frei) in Gleitkomma: der exakte Prüfweg für kleine N, geschrieben ohne Bezug auf msg_qp."""
    import numpy as np

    H = np.asarray(H, float)
    f = np.asarray(f, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
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
    return best, bestv


@pytest.fixture
def wagen_hand_case():
    """Das Handbeispiel: Wagen, N = 2, Q = P = I, R = 1, |u| ≤ 1/2, x0 = (7, −2). Von Hand: H = [[8, 2], [2, 4]], f = (−2, −4), unbeschränkt (0, 1), MPC (1/8, 1/2),
    abgeschnitten (0, 1/2); Kosten 93, 1495/16, 187/2."""
    return {"x0": (Fr(7), Fr(-2)), "R": Fr(1), "P": [[Fr(1), Fr(0)], [Fr(0), Fr(1)]], "umax": Fr(1, 2)}
