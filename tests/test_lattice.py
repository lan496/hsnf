import numpy as np

from hsnf.lattice import compute_dual, compute_intersection, compute_union, equivalent


def test_equivalence():
    lattice1 = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    lattice1_2 = np.array([[1, 5, 0], [1, 6, 0]])
    assert equivalent(lattice1, lattice1_2)

    lattice2 = np.array(
        [
            [1, 5, 0],
            [1, 1, 0],
        ]
    )
    assert not equivalent(lattice1, lattice2)


def test_union():
    lattice1 = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
        ]
    )
    lattice2 = np.array(
        [
            [1, 0, 2],
            [1, 2, 1],
        ]
    )
    actual = compute_union(lattice1, lattice2)
    expect = np.diag([1, 1, 1])
    assert np.allclose(actual, expect)


def test_dual():
    A = np.array([[6, 4, 10], [-1, 1, -5]])
    B = compute_dual(A)
    assert np.allclose(B @ A.T, np.eye(2))


def test_intersection():
    lattice1 = np.diag([1, 1, 1])
    lattice2 = np.array(
        [
            [-3, 4, 0],
            [-4, -3, 0],
            [0, 0, 5],
        ]
    )
    actual = compute_intersection(lattice1, lattice2)
    expect = np.array(
        [
            [25, 0, 0],
            [-7, 1, 0],
            [0, 0, 5],
        ]
    )
    assert np.allclose(actual, expect)
