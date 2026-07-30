"""Tiny pure-Python linear algebra — no NumPy, deterministic.

Just enough dense matrix machinery for the RAIM/ARAIM least-squares position
solver and the multi-constellation cross-checks: transpose, matmul, a
partial-pivot Gaussian solver, and a normal-equations least-squares. Matrices
are plain ``list[list[float]]`` (row-major); vectors are ``list[float]``.
Kept small and readable rather than fast — GNSS design matrices are tiny
(a dozen rows, a handful of columns).
"""

from __future__ import annotations

import math


def shape(A) -> tuple:
    return (len(A), len(A[0]) if A else 0)


def transpose(A):
    return [list(col) for col in zip(*A)] if A else []


def matmul(A, B):
    if not A or not B:
        return []
    if len(A[0]) != len(B):
        raise ValueError(f"matmul shape mismatch {shape(A)} x {shape(B)}")
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def matvec(A, x):
    if len(A[0]) != len(x):
        raise ValueError(f"matvec shape mismatch {shape(A)} x {len(x)}")
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def solve(A, b):
    """Solve the square system ``A x = b`` by Gaussian elimination w/ partial pivot."""
    n = len(A)
    if any(len(row) != n for row in A) or len(b) != n:
        raise ValueError("solve expects a square system")
    # augmented copy
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        # partial pivot: largest magnitude in/under the diagonal
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            raise ValueError("singular matrix")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / pv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def lstsq(A, b):
    """Least-squares solution of (possibly overdetermined) ``A x = b``.

    Solves the normal equations ``AᵀA x = Aᵀb``. Returns ``(x, residuals)``
    where ``residuals = b - A x``. Raises on a rank-deficient geometry.
    """
    At = transpose(A)
    AtA = matmul(At, A)
    Atb = matvec(At, b)
    x = solve(AtA, Atb)
    pred = matvec(A, x)
    resid = [bi - pi for bi, pi in zip(b, pred)]
    return x, resid


def vnorm(v) -> float:
    return math.sqrt(sum(c * c for c in v))


def vsub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]


def vadd(a, b):
    return [ai + bi for ai, bi in zip(a, b)]
