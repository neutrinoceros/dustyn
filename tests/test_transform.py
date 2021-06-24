import numpy as np
import pytest

from dustyn.core.transform import spherical2cartesian


@pytest.mark.parametrize(
    "coords,expected",
    [
        (
            np.array([1, np.pi / 2, np.pi / 2], dtype="float64"),
            np.array([0, 1, 0], dtype="float64"),
        ),
        (
            np.array([1, np.pi / 2, 0], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
        ),
        (
            np.array([1, np.pi / 2, np.pi / 4], dtype="float64"),
            np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype="float64"),
        ),
        (
            np.array(
                [np.sqrt(3), np.pi / 2 - np.arctan(1 / np.sqrt(2)), np.pi / 4],
                dtype="float64",
            ),
            np.ones(3, dtype="float64"),
        ),
    ],
)
def test_spherical_to_cartesian_transform_coords(coords, expected):
    res = spherical2cartesian(position=coords)
    np.testing.assert_array_almost_equal(res, expected, decimal=16)

    flipped_coords = coords.copy()
    flipped_coords[1] = np.pi - flipped_coords[1]
    res2 = spherical2cartesian(position=flipped_coords)

    flipped_res = res.copy()
    flipped_res[2] *= -1
    np.testing.assert_array_almost_equal(res2, flipped_res, decimal=16)


@pytest.mark.parametrize(
    "coords,velocities,expected",
    [
        (
            np.array([1, np.pi / 2, 0], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
        ),
        (
            np.array([1, np.pi / 2, np.pi / 2], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
            np.array([0, 1, 0], dtype="float64"),
        ),
        (
            np.array([1, np.pi / 2, np.pi], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
            np.array([-1, 0, 0], dtype="float64"),
        ),
        (
            np.array([1, np.pi / 2, 3 * np.pi / 2], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
            np.array([0, -1, 0], dtype="float64"),
        ),
        (
            np.array([1, np.pi / 2, 0], dtype="float64"),
            np.array([0, 1, 0], dtype="float64"),
            np.array([0, 0, -1], dtype="float64"),
        ),
        (
            np.array([1, np.pi / 2, 0], dtype="float64"),
            np.array([0, 0, 1], dtype="float64"),
            np.array([0, 1, 0], dtype="float64"),
        ),
    ],
)
def test_spherical_to_cartesian_transform_vels(coords, velocities, expected):
    res = spherical2cartesian(position=coords, vector=velocities)
    np.testing.assert_array_almost_equal(res, expected, decimal=15)

    res2 = spherical2cartesian(position=coords, vector=-velocities)
    np.testing.assert_array_almost_equal(res2, -expected, decimal=15)
