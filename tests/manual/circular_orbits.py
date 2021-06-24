import matplotlib.pyplot as plt
import numpy as np

from dustyn.core.integrate import RK4
from dustyn.core.newton import NewtonSpherical3D
from dustyn.plot.trajectory import Projected3DTrajectoryPlot

model = NewtonSpherical3D()
initial_states = [
    # a particle in a circular orbit with axis y
    np.array([0, 1, 0, 1, np.pi / 2, 0], dtype="float64"),
    # a particle in a circular orbit with axis z
    np.array([0, 0, 1, 1, np.pi / 2, 0], dtype="float64"),
    np.array([0, np.sqrt(0.5), np.sqrt(0.5), 1, np.pi / 2, 0], dtype="float64"),
]

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(18, 8))

for R0, axcol in zip(initial_states, axes.T):
    record = RK4.integrate(model, R0, tstart=0, tstop=100, CFL=0.3)

    for normal, ax in zip("yz", axcol):
        Projected3DTrajectoryPlot(
            ax,
            fig,
            y=record.states[:, 3:],
            normal=normal,
            input_geometry="spherical",
        )
        ax.set(xlim=(-2, 2), ylim=(-2, 2))

fig.savefig("/tmp/circular_orbits.png")
