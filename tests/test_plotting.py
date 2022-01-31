import matplotlib.pyplot as plt
import numpy as np
import pytest
import requests
import sympy as sp
from matplotlib.colors import TwoSlopeNorm
from sympy.abc import phi
from sympy.abc import r
from sympy.abc import theta

from dustyn.core.integrate import RK4
from dustyn.core.medium import AnalyticMedium
from dustyn.core.newton import NewtonSpherical3D
from dustyn.core.sampling import SpaceSampler
from dustyn.plot.background import SphereSlicePlot
from dustyn.plot.record_plot import RecordPlot
from dustyn.plot.trajectory import Projected3DTrajectoryPlot
from dustyn.sswind_models import SSWDisk
from dustyn.sswind_models import SSWMedium

mpl_compare = pytest.mark.mpl_image_compare(
    savefig_kwargs={"bbox_inches": "tight", "dpi": 80},
    style="default",
)

# in ALL of the following, axes are consistently ordered `theta, r`
# because that's how polar projections are implemented in matplotlib


# define arbitrary callbacks
def annotate_scale_height(ax, fig, r_grid, theta_grid, medium) -> None:
    ax.contour(
        theta_grid,
        r_grid,
        np.abs(
            medium.get("z", r_grid, theta_grid, 0)
            / medium.get("scale_height", r_grid, theta_grid, 0)
        ),
        levels=[1, 2, 3],
        colors="black",
        linestyles=["-", "--", ":"],
        alpha=0.6,
    )


def annotate_lic(ax, fig, r_grid, theta_grid, medium) -> None:
    try:
        from lic import lic
    except ImportError:
        return
    z = lic(
        medium.get("velocity_theta", r_grid, theta_grid, 0),
        medium.get("velocity_r", r_grid, theta_grid, 0),
    )
    ax.pcolormesh(theta_grid, r_grid, z, cmap="gray", shading="gouraud")


@mpl_compare
def test_disk_drawning():
    # define the grid
    offset = np.pi / 3
    bounds = [0.2, 1, 0 + offset, np.pi - offset]
    grid = SpaceSampler()(bounds=bounds, npoints=(100, 100))

    # define the medium
    z_expr = r * sp.cos(theta)
    H_expr = 0.1 * r**1.25

    # The model is 3D axisymetric: we require a third component (phi) but don't use it
    medium = AnalyticMedium(
        symbols=[r, theta, phi],
        defs=dict(
            density=r**-1 * sp.exp(-(z_expr**2 / (2 * H_expr**2))),
            velocity_r=-1 / (10 + theta % sp.pi - sp.pi / 2),
            velocity_phi=r ** -sp.Rational(3, 2),
            velocity_theta=0,
            scale_height=H_expr,
            z=z_expr,
        ),
    )

    # finally, setup the figure
    fig = plt.figure(figsize=(10, 6))
    axd = fig.add_subplot(131, projection="polar")
    axd.set(title="poloidal density")
    axv = fig.add_subplot(132, projection="polar")
    axv.set(title="poloidal velocity")
    axz = fig.add_subplot(133, projection="polar")
    axz.set(title="midplane density")

    # create the background
    ssp = SphereSlicePlot(
        axd,
        fig,
        grid=grid,
        medium=medium,
        background=medium.get("density", *grid, 0),
        zlabel=r"\rho",
        zscale="log",
        cmap="inferno",
        levels=50,
        cbar_orientation="horizontal",
    )

    # add callbacks
    ssp.callbacks(
        annotate_scale_height,
    )

    # secondary example with no background and a lic-based callback
    ssp = SphereSlicePlot(
        axv,
        fig,
        medium=medium,
        grid=grid,
        background=None,
    )
    ssp.callbacks(
        annotate_scale_height,
        annotate_lic,
    )

    # third, vertical projection
    bounds = [0.2, 1, 0, 2 * np.pi]
    grid = rg, phig = SpaceSampler()(bounds=bounds, npoints=(10, 100))
    ssp = SphereSlicePlot(
        axz,
        fig,
        grid=grid,
        medium=medium,
        background=medium.get("density", rg, np.pi / 2, phig),
        zlabel=r"\rho",
        cmap="inferno",
        levels=50,
        cbar_orientation="horizontal",
    )

    for ax in (axd, axv, axz):
        ax.set(yticks=())

    return fig
    # fig.savefig("/tmp/sampler_medium_disk.jpg", bbox_inches="tight", dpi=500)


@pytest.fixture()
def ppdwind_example_file(tmp_path):
    sample_url = "https://raw.githubusercontent.com/glesur/PPDwind/main/beta%3D1.0e%2B04-Am%3D0.25-Rm%3Dinf.dat"
    answer = requests.get(sample_url)
    if not answer.ok:
        raise RuntimeError
    storage_file = tmp_path.joinpath("model.dat")
    storage_file.write_text(answer.text)
    return storage_file


@pytest.fixture()
def trajectory_record(ppdwind_example_file):
    medium = SSWMedium(ppdwind_example_file)
    model = SSWDisk(medium=medium, mass_to_surface=3e-6)

    R0 = np.array([0, 0, 0, 1, np.pi / 4 + 0.1, np.pi / 2])
    record = RK4.integrate(model, R0, tstart=0, tstop=1_000, CFL=0.3, pbar=False)
    return record


@mpl_compare
def test_raw_record_plot(trajectory_record):
    fig, axes = plt.subplots(
        ncols=3, nrows=3, figsize=(12, 10), sharex=True, sharey=True
    )
    for colors, row in zip((None, "normal", "time"), axes):
        for normal, ax in zip("xyz", row):
            RecordPlot(trajectory_record, ax, normal=normal, colors=colors)

    return fig


@mpl_compare
def test_poloidal_record_plot(trajectory_record):
    fig, axes = plt.subplots(ncols=3, figsize=(25, 4))
    for colors, cmap, ax in zip(
        (None, "normal", "time"), (None, "twilight", "viridis"), axes
    ):
        kwargs = {
            "lw": 4,
        }
        if cmap is not None:
            kwargs["cmap"] = cmap
        RecordPlot(
            trajectory_record,
            ax,
            geometry="cylindrical",
            colors=colors,
            normal="-phi",
            **kwargs,
        )
    fig.suptitle("poloidal trajectory plots")
    return fig


@mpl_compare
def test_sswind_backgroundplot(ppdwind_example_file):
    medium = SSWMedium(ppdwind_example_file)

    theta_bounds = medium.ds["theta"][0], medium.ds["theta"][-1]
    # define the grid
    bounds = [0.2, 1, *theta_bounds]
    rg, thetag = grid = SpaceSampler()(bounds=bounds, npoints=(30, 100))

    # finally, setup the figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(121, projection="polar")

    # create the background
    SphereSlicePlot(
        ax,
        fig,
        grid=grid,
        medium=medium,
        background=medium.get("sound_speed", rg, thetag, 0),
        zlabel=r"c_\mathrm{s}",
        cmap="inferno",
        levels=100,
        cbar_orientation="horizontal",
    )
    ax.set(yticks=())

    ax = fig.add_subplot(122, projection="polar")
    SphereSlicePlot(
        ax,
        fig,
        grid=grid,
        medium=medium,
        background=medium.get("density", rg, thetag, 0),
        zlabel=r"\rho",
        cmap="RdBu",
        levels=100,
        zscale="log",
        cbar_orientation="horizontal",
    )

    ax.set(yticks=())
    return fig


# the following tests are not really image comparison tests but
# the pytest-mpl plugin provides an easy way to keep a reference for integration tests...
@mpl_compare
def test_circular_orbits():
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
    return fig


@mpl_compare
def test_angular_momentum_conservation():
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
        [(None, None, TwoSlopeNorm(0), "RdBu"), (record.times, "t", None, "Greys")],
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
    return fig
