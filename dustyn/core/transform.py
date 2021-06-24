from typing import Any, Callable, Optional

import numpy as np
from numpy import ScalarType, cos, sin


def _get_rotation_matrix(r: ScalarType, theta: ScalarType, phi: ScalarType):
    return np.array(
        [
            [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)],
            [cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)],
            [-sin(phi), cos(phi), 0],
        ],
    )


def spherical2cartesian(
    position: np.ndarray, vector: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Transform a spherical `position` (r, theta, phi) coordinate array
    to cartesian (x, y, z).

    When `vector` is provided, transform this instead.
    """
    assert 3 in position.shape
    if len(position) != 3:
        position = position.T
    r, theta, phi = position

    if isinstance(r, ScalarType):
        MROT = _get_rotation_matrix(*position)
        if vector is None:
            return [r, 0, 0] @ MROT
        return vector @ MROT

    assert r.ndim == 1

    ret = np.empty_like(position, order="F")
    for i in range(len(r)):
        if vector is not None:
            v = vector[:, i]
        else:
            v = None
        ret[:, i] = spherical2cartesian(position=position[:, i], vector=v)
    return ret


def spherical2cylindrical(position: np.ndarray) -> np.ndarray:
    assert position.ndim == 2
    assert 3 in position.shape
    if len(position) != 3:
        position = position.T
    r, theta, phi = position
    return np.array([r * np.sin(theta), phi, r * np.cos(theta)])


def cylindrical2spherical(position: np.ndarray) -> np.ndarray:
    assert 3 in position.shape
    if len(position) != 3:
        position = position.T
    R, phi, z = position
    return np.array([np.sqrt(R ** 2 + z ** 2), np.arctan2(R, z), phi])


# this is meant to be imported from outside
TRANSFORMATIONS: dict[str, dict[str, Callable[[Any], np.ndarray]]] = {
    "spherical": {
        "cartesian": spherical2cartesian,
        "cylindrical": spherical2cylindrical,
    },
    "cylindrical": {
        "spherical": cylindrical2spherical,
    },
}
