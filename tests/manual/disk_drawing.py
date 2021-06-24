import cmocean  # noqa F401
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.abc import phi, r, theta

from dustyn.core.medium import AnalyticMedium
from dustyn.core.sampling import SpaceSampler
from dustyn.plot.background import SphereSlicePlot

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


if __name__ == "__main__":

    # define the grid
    offset = np.pi / 3
    bounds = [0.2, 1, 0 + offset, np.pi - offset]
    grid = SpaceSampler()(bounds=bounds, npoints=(100, 100))

    # define the medium
    z_expr = r * sp.cos(theta)
    H_expr = 0.1 * r ** 1.25

    # The model is 3D axisymetric: we require a third component (phi) but don't use it
    medium = AnalyticMedium(
        symbols=[r, theta, phi],
        defs=dict(
            density=r ** -1 * sp.exp(-(z_expr ** 2 / (2 * H_expr ** 2))),
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
        cmap="cmo.thermal",
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
        cmap="cmo.thermal",
        levels=50,
        cbar_orientation="horizontal",
    )

    for ax in (axd, axv, axz):
        ax.set(yticks=())

    fig.savefig("/tmp/sampler_medium_disk.jpg", bbox_inches="tight", dpi=500)
