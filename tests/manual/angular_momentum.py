from pathlib import Path

import cmasher  # noqa: F401
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from dustyn.core.integrate import RK4
from dustyn.core.newton import NewtonSpherical3D
from dustyn.plot.trajectory import Projected3DTrajectoryPlot

sty = str(
    Path(__file__).parents[3].resolve().joinpath("notebooks", "robert+2020.mplstyle")
)
mpl.rc_file(sty)

model = NewtonSpherical3D()
R0 = np.array([0.5, 0.2, 1, 1, np.pi / 2, 0], dtype="float64")
record = RK4.integrate(model, R0, tstart=0, tstop=20, CFL=0.3)

fig, axes = plt.subplots(ncols=3, figsize=(18, 5))

L = model.angular_momentum(record.states.T)
for Lx, label in zip(L, "xyz"):
    axes[0].plot(record.times, Lx, label=r"$L_%s$" % label)
axes[0].set(
    title="angular momentum components (cartesian)", xlabel=r"$t$", ylabel=r"$L$"
)
axes[0].legend()


for (colors, zlabel, norm, cmap), ax in zip(
    [(None, None, TwoSlopeNorm(0), "cmr.wildfire"), (record.times, "t", None, "Greys")],
    axes[1:],
):
    Projected3DTrajectoryPlot(
        ax,
        fig,
        record.states.T[3:],
        normal="z",
        input_geometry="spherical",
        colors=colors,
        zlabel=zlabel,
        cmap=cmap,
        norm=norm,
    )
    ax.scatter(0, 0, marker="*", color="k")

fig.savefig("/tmp/angmom.png", bbox_inches="tight")
