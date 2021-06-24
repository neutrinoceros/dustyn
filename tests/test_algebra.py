import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from dustyn.core.algebra import cross_product


@given(arrays(dtype="float64", shape=3, elements=floats(-1e4, 1e4)))
def test_cross_product_colinear_vectors(R):
    colinR = np.random.randn() * R
    expected = np.zeros(3, dtype="float64")
    result1 = cross_product(colinR, R)
    result2 = cross_product(R, colinR)
    np.testing.assert_almost_equal(result1, expected, decimal=4)
    np.testing.assert_almost_equal(result2, expected, decimal=4)

    np.testing.assert_array_equal(result1, np.cross(colinR, R))


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            np.array([1, 0, 0], dtype="float64"),
            np.array([0, 1, 0], dtype="float64"),
            np.array([0, 0, 1], dtype="float64"),
        ),
        (
            np.array([0, 0, 1], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
            np.array([0, 1, 0], dtype="float64"),
        ),
        (
            np.array([0, 1, 0], dtype="float64"),
            np.array([0, 0, 1], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
        ),
    ],
)
def test_cross_product_basics(a, b, expected):
    np.testing.assert_array_equal(cross_product(a, b), expected)
    np.testing.assert_array_equal(cross_product(b, a), -expected)
    np.testing.assert_array_equal(cross_product(a, b), np.cross(a, b))
