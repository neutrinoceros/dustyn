import numpy as np


def cross_product(a: np.ndarray, b: np.ndarray, /) -> np.ndarray:
    """
    This implementation is 10x faster than np.cross in my applications.
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )
