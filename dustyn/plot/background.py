import inspect
import weakref
from typing import Callable, Optional

import numpy as np
from more_itertools import always_iterable
from sympy import pi as SPI

from dustyn.core.medium import Medium


class SphereSlicePlot:
    def __init__(
        self,
        ax,
        fig,
        *,
        grid: np.ndarray,
        medium: Medium,
        background: Optional[np.ndarray] = None,
        zscale="lin",
        zlabel: Optional[str] = None,
        cbar_orientation: str = "horizontal",
        **kwargs,
    ):
        # note that I'm providing a "simple" api to set logscale in the z direction
        # because I could not find a satisfying way to do it with a single argument
        # to mpl's api. It will not work with negative values (no symlog implementation for now)

        if cbar_orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"Unknown value for `{cbar_orientation=}` (expected 'horizontal' or 'vertical')."
            )
        if zscale not in ("lin", "log"):
            raise ValueError(
                f"Unknown value for `{zscale=}` (expected 'lin' or 'log')."
            )

        # grid is assumed to be written in the canonical order (r, theta)
        # but matplotlib only handles polar projections with
        # "x" as the angular coordinate, so we unpack here and will use those components
        # in a reversed order
        self.rg, self.thetag = grid
        self.fig = weakref.proxy(fig)
        self.ax = weakref.proxy(ax)
        self.medium = weakref.proxy(medium)

        ax.set(xlim=(self.thetag[0, 0], self.thetag[-1, -1]))
        self._setup_curvilinear_axis()

        if background is None:
            return

        if zscale == "log":
            background = np.log10(background)

        im = ax.contourf(self.thetag, self.rg, background, **kwargs)

        cbar_kwargs = {"orientation": cbar_orientation}
        if zlabel is not None:
            cbar_kwargs["label"] = r"$%s$" % zlabel
        cbar = fig.colorbar(im, ax=ax, **cbar_kwargs)
        if cbar_kwargs["orientation"] == "horizontal":
            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
            axis = cbar.ax.xaxis
        else:
            axis = cbar.ax.yaxis

        if zscale == "log":
            levels = range(
                int(np.ceil(background.min())), int(np.ceil(background.max()))
            )
            axis.set_ticks(levels)
            axis.set_ticklabels(["$10^{%d}$" % level for level in levels])

    def _setup_curvilinear_axis(self, div: int = 6):
        if self.ax.name == "polar":
            # this is relevant only if the x coordinate is the latitude (theta) in spherical
            self.ax.set(
                rorigin=0,
                theta_zero_location="N",  # theta = 0 is on the vertical axis
                theta_direction=-1,  # set clock-wise counting
            )
            self.ax.grid(False)

        # the x bounds are actually already stored in radian
        # only the xticklabels are written in degrees...
        int_bounds = np.ceil(np.array(self.ax.get_xlim()) / (np.pi / div)).astype(
            "int64"
        )
        ticks_mul = np.arange(int_bounds[0], int_bounds[1])

        if ticks_mul[-1] - ticks_mul[0] >= div * 2:
            # remove the last label if it is redundant to avoid overlaps
            ticks_mul = ticks_mul[:-1]
        self.ax.set_xticks(np.pi / div * ticks_mul)

        labels = [
            r"$%s$" % str(int(m) * SPI / div).replace("pi", r"\pi").replace("*", "")
            for m in ticks_mul
        ]
        self.ax.set_xticklabels(labels)

    def callbacks(self, *callbacks: Callable):
        sig = ["ax", "fig", "r_grid", "theta_grid", "medium"]
        for cb in always_iterable(callbacks):
            params = list(inspect.signature(cb).parameters)
            if params != sig:
                raise TypeError(
                    f"Received callback {cb} with signature {params}, expected {sig}."
                )
            cb(self.ax, self.fig, self.rg, self.thetag, self.medium)
