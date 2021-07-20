import weakref
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dustyn.core.record import Record
from dustyn.core.transform import TRANSFORMATIONS

CURVILINEAR_AXES = frozenset({"phi", "theta"})
AXES: dict[str, list[str]] = {
    "cartesian": ["x", "y", "z"],
    "cylindrical": ["R", "phi", "z"],
    "spherical": ["r", "theta", "phi"],
}
LATEX_SYMBOLS: dict[str, str] = {
    "phi": r"$\varphi$",
    "theta": r"$\theta$",
}


def texify_symbol(s: str) -> str:
    return LATEX_SYMBOLS.get(s, s)


@dataclass
class Vector:
    data: np.ndarray
    label: str

    def __len__(self):
        # NOTE: this could be abstracted into a DataContainer abstract dataclass
        # since it's common to the Record class
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape


class RecordPlot:
    def __init__(
        self,
        record: Record,
        ax: Optional[Axes] = None,
        *,
        geometry: str = "cartesian",
        normal: Optional[str] = None,
        clabel: Optional[str] = None,
        colors: Optional[Union[np.ndarray, str]] = None,
        cbar: bool = True,
        logscale: bool = False,
        L0=1.0,
        **kwargs,
    ):
        if ax is None:
            _, ax = plt.subplots()
            self.ax = ax
        else:
            self.ax = weakref.proxy(ax)
        self.record = weakref.proxy(record)
        self.geometry = geometry
        self.axes_labels = AXES[self.geometry]
        normal = normal or AXES[self.geometry][2]
        self._flip_axes = normal.startswith("-")
        if self._flip_axes:
            normal = normal[1:]
        self.normal = normal
        self.L0 = L0

        self._setup_transform()

        x1, x2 = self._get_plane_vectors()

        x3: Optional[Vector]
        if isinstance(colors, str):
            if colors == "normal":
                x3 = self._get_normal_vector()
            elif colors == "time":
                x3 = Vector(data=record.times, label=r"$t$")
            else:
                raise ValueError
        elif isinstance(colors, np.ndarray):
            x3 = Vector(data=colors, label=clabel or "")
        elif colors is None:
            x3 = None
        else:
            raise TypeError

        self._setup_plot(x1, x2, x3, **kwargs)

        if logscale:
            if x1.label not in CURVILINEAR_AXES:
                ax.set_xscale("log")
            if x2.label not in CURVILINEAR_AXES:
                ax.set_yscale("log")

        if cbar and colors is not None:
            self.setup_cbar()

    def _setup_transform(self) -> None:
        record_geometry = self.record.metadata["geometry"]
        if self.geometry == record_geometry:
            self.states = self.record.states[:, 3:].T
        else:
            try:
                transform = TRANSFORMATIONS[record_geometry][self.geometry]
            except KeyError as err:
                raise NotImplementedError from err
            self.states = transform(self.record.states[:, 3:])

    def normalize_data(self, axis_name: str) -> Vector:
        data = self.states[self.axes_labels.index(axis_name)]
        if axis_name in CURVILINEAR_AXES:
            data %= 2 * np.pi
        else:
            data *= self.L0
        return Vector(data, label=axis_name)

    def _get_plane_vectors(self) -> tuple[Vector, Vector]:
        names = deque(self.axes_labels)
        while names[-1] != self.normal:
            names.rotate()
        labels = list(names)[:2]
        if self._flip_axes:
            labels = labels[::-1]
        # deactivating mypy here because it seems to be insecure about the return
        # tuple length, but it should be obvious from the code above that
        # it will always be 2 and match the signature
        return tuple(self.normalize_data(label) for label in labels)  # type: ignore

    def _get_normal_vector(self) -> Vector:
        if not (label := self.normal) in self.axes_labels:
            raise ValueError(
                f"Unknown normal axis '{self.normal}' with geometry '{self.geometry}'."
            )
        return self.normalize_data(label)

    def set(self, **kwargs):
        self.ax.set(**kwargs)

    def _setup_plot(
        self,
        x1: Vector,
        x2: Vector,
        x3: Optional[Vector],
        **kwargs,
    ) -> None:

        self.set(xlabel=texify_symbol(x1.label), ylabel=texify_symbol(x2.label))
        if not any(label in CURVILINEAR_AXES for label in (x1.label, x2.label)):
            self.set(aspect="equal")

        if x3 is None:
            self.ax.plot(x1.data, x2.data, **kwargs)
            return

        # this is a workaround to compensate for the fact that
        # LineCollection doesn't force xlim and ylim to auto adjust,
        # so we'll draw a basic "plot" in even if using colors,
        # but we'll make it invisible
        self.ax.plot(x1.data, x2.data, lw=0)

        points = np.array([x1.data, x2.data]).T.reshape(-1, 1, 2)

        # can probably be done in a cleaner way
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # this is adapted from
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

        # Create a continuous norm to map from data points to colors
        lc = LineCollection(segments, **kwargs)
        # Set the values used for colormapping
        lc.set_array(x3.data)
        lc.set_linewidth(kwargs.get("lw", 2))
        self._mappable = self.ax.add_collection(lc)
        self.x3 = x3

    def setup_cbar(self):
        # Controls that the colorbar fits the size of the plot

        # NOTE: I can't seem to find one method that works well in every situation (log/lin).
        # Additional difficulties arise when the size of the figure is updated after the cbar
        # was setup, so the recommended way to use this is to run any call to
        # ax.set_xlim, ax.set_ylim, fig.subplots_adjust...
        # ahead of plotting, and *then* call this method. This is why it is separated from the
        # _setup_plot method, and also why it is written as user-facing.

        if self.ax.get_yscale() == "log":
            axpos = self.ax.get_position()
            # args are:
            # left, bottom, width, height
            cax = self.ax.figure.add_axes(
                [axpos.x1, axpos.y0, (axpos.x1 - axpos.x0) * 0.05, axpos.height]
            )
        else:
            # this works extremely poorly with logscale
            # it is adapted from
            # https://stackoverflow.com/questions/29516157/set-equal-aspect-in-plot-with-colorbar
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0)

        self.ax.figure.colorbar(
            self._mappable, cax=cax, label=texify_symbol(self.x3.label)
        )
