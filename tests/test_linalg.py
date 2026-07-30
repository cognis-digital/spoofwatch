import math

import pytest

from spoofwatch import linalg


def test_transpose():
    assert linalg.transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_transpose_empty():
    assert linalg.transpose([]) == []


def test_matmul_identity():
    A = [[1.0, 2.0], [3.0, 4.0]]
    I = linalg.identity(2)
    assert linalg.matmul(A, I) == A


def test_matmul_values():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    assert linalg.matmul(A, B) == [[19, 22], [43, 50]]


def test_matmul_shape_mismatch():
    with pytest.raises(ValueError):
        linalg.matmul([[1, 2]], [[1, 2]])


def test_matvec():
    assert linalg.matvec([[1, 2], [3, 4]], [1, 1]) == [3, 7]


def test_matvec_mismatch():
    with pytest.raises(ValueError):
        linalg.matvec([[1, 2]], [1, 2, 3])


def test_identity():
    assert linalg.identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_solve_simple():
    # x + y = 3 ; x - y = 1  -> x=2, y=1
    x = linalg.solve([[1, 1], [1, -1]], [3, 1])
    assert math.isclose(x[0], 2.0, abs_tol=1e-9)
    assert math.isclose(x[1], 1.0, abs_tol=1e-9)


def test_solve_3x3():
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    x = linalg.solve(A, b)
    assert all(math.isclose(a, e, abs_tol=1e-9) for a, e in zip(x, [2, 3, -1]))


def test_solve_singular_raises():
    with pytest.raises(ValueError):
        linalg.solve([[1, 2], [2, 4]], [1, 2])


def test_solve_needs_square():
    with pytest.raises(ValueError):
        linalg.solve([[1, 2, 3]], [1])


def test_lstsq_exact():
    A = [[1, 0], [0, 1], [1, 1]]
    b = [1, 2, 3]   # consistent -> residual ~ 0
    x, resid = linalg.lstsq(A, b)
    assert all(math.isclose(r, 0.0, abs_tol=1e-9) for r in resid)
    assert math.isclose(x[0], 1.0, abs_tol=1e-9)
    assert math.isclose(x[1], 2.0, abs_tol=1e-9)


def test_lstsq_overdetermined_line_fit():
    # fit y = a*x + b through (0,0),(1,1),(2,2),(3,3.0) -> a=1,b=0
    A = [[0, 1], [1, 1], [2, 1], [3, 1]]
    b = [0, 1, 2, 3]
    x, resid = linalg.lstsq(A, b)
    assert math.isclose(x[0], 1.0, abs_tol=1e-6)
    assert math.isclose(x[1], 0.0, abs_tol=1e-6)


def test_vnorm():
    assert math.isclose(linalg.vnorm([3, 4]), 5.0)


def test_vsub_vadd():
    assert linalg.vsub([5, 7], [1, 2]) == [4, 5]
    assert linalg.vadd([1, 2], [3, 4]) == [4, 6]


def test_shape():
    assert linalg.shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
