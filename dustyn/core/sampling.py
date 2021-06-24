from typing import Any, Optional, Sequence, Tuple

import numpy as np

from dustyn._typing import SingleOrDouble


class SpaceSampler:

    """Create a 2D discrete rectilinear grid specified by bounds, geometry and spacing type."""

    @staticmethod
    def get_grid(
        *,
        bounds: np.ndarray,
        npoints: Tuple[int, int] = (100, 100),
        spacing: Tuple[str, str] = ("linear", "linear"),
    ) -> np.ndarray:

        if len(bounds) != 4:
            raise ValueError(f"Expected a 4-size `bounds` array, received `{bounds}`.")

        if spacing != ("linear", "linear"):
            raise NotImplementedError

        if not isinstance(spacing, Sequence):
            raise TypeError(f"Expected a sequence, received `{type(spacing)}`.")

        if spacing == ("linear", "linear"):
            space_discretization = np.linspace

        xv = space_discretization(*bounds[0:2], npoints[0])
        yv = space_discretization(*bounds[2:4], npoints[1])

        return np.meshgrid(xv, yv)

    def __call__(
        self,
        *,
        bounds: Optional[np.ndarray] = None,
        npoints: SingleOrDouble[int] = 100,
        ax: Optional[Any] = None,
        geometry: str = "cartesian",
        spacing: SingleOrDouble[str] = "linear",
    ) -> np.ndarray:

        if geometry != "cartesian":
            raise NotImplementedError

        if bounds is None:
            if geometry != "cartesian":
                raise TypeError(
                    "`bounds` keyword argument is required with `geometry` != 'cartesian'."
                )
            if ax is None:
                raise TypeError(
                    "Either `bounds` or `ax` keyword argument must be specified."
                )

            bounds = [*ax.get_xlim(), *ax.get_ylim()]

        if isinstance(npoints, int):
            npoints = (npoints, npoints)
        if isinstance(spacing, str):
            spacing = (spacing, spacing)
        return self.get_grid(bounds=bounds, npoints=npoints, spacing=spacing)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sampler = SpaceSampler()
    grid = sampler(bounds=[-10, 1, 1, 10], npoints=10)
    data = np.random.randn(*grid[0].shape)
    fig, ax = plt.subplots()
    ax.contourf(*grid, data)
    fig.savefig("/tmp/sampler.jpg")
