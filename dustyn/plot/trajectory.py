import warnings
from collections import deque
from typing import Optional

import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dustyn.core.transform import spherical2cartesian

NAME2COL = {"x": 0, "y": 1, "z": 2}


class Projected3DTrajectoryPlot:
    """
    Plot a 3D trajectory, a represented in a Cartesian coordinate system,
    in 2 dimension. The normal coordinate is shown in color.
    """

    def __init__(
        self,
        ax,
        fig,
        y,
        normal: str = "z",
        input_geometry="cartesian",
        **kwargs,
    ):

        if input_geometry == "spherical":
            y = spherical2cartesian(position=y).T
        elif input_geometry == "cartesian":
            pass
        else:
            raise ValueError(f"Unrecognized value `{input_geometry=}`.")

        axes_names = deque("xyz")
        if normal not in axes_names:
            raise ValueError

        while not axes_names[-1] == normal:
            axes_names.rotate()

        labels = list(axes_names)[:2]
        ax.set(xlabel=r"$%s$" % labels[0], ylabel=r"$%s$" % labels[1], aspect="equal")
        x1, x2 = (y[:, NAME2COL[v]] for v in labels)

        self.setup_plot(ax, fig, y, x1, x2, normal, **kwargs)

    def setup_plot(
        self,
        ax,
        fig,
        y,
        x1,
        x2,
        normal: str,
        colors: Optional[np.ndarray] = None,
        zlabel: Optional[str] = None,
        **kwargs,
    ):
        if colors is None:
            x3 = y[:, NAME2COL[normal]]
            if zlabel is not None:
                warnings.warn(
                    "Received `zlabel` but `colors` keyword argument is unspecified, "
                    "`zlabel` will be ignored."
                )
            zlabel = normal
        else:
            x3 = colors

        points = np.array([x1, x2]).T.reshape(-1, 1, 2)

        # can probably be done in a cleaner way
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # this is adapted from
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

        # Create a continuous norm to map from data points to colors
        lc = LineCollection(segments, **kwargs)
        # Set the values used for colormapping
        lc.set_array(x3)
        lc.set_linewidth(kwargs.get("lw", 2))
        line = ax.add_collection(lc)

        # this part controls that the colorbar fits the size of the plot
        # adapted from
        # https://stackoverflow.com/questions/29516157/set-equal-aspect-in-plot-with-colorbar
        divider = make_axes_locatable(ax)
        color_axis = divider.append_axes("right", size="5%", pad=0)

        cbar_kwargs = dict(cax=color_axis, pad=0)
        if zlabel is not None:
            # this is a workaround a bug in matplotlib
            cbar_kwargs.update(dict(label=r"$%s$" % zlabel))
        fig.colorbar(line, **cbar_kwargs)


class SimpleProjected3DTrajectoryPlot(Projected3DTrajectoryPlot):
    def setup_plot(self, ax, fig, y, x1, x2, normal, **kwargs):
        ax.plot(x1, x2, **kwargs)
