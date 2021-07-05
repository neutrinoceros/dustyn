import matplotlib.pyplot as plt
import numpy as np
import pytest

from dustyn.core.sampling import SpaceSampler


# this should be done via hypothesis...
@pytest.fixture()
def random_bounds():
    return (sorted(np.random.randn(2)) for _ in range(2))


def test_sampler_singleton():
    s1 = SpaceSampler()
    s2 = SpaceSampler()
    assert s2 is s1


@pytest.mark.parametrize(
    "npoints,expected_shape",
    [
        (5, (5, 5)),
        ((10, 5), (5, 10)),
        ((5, 10), (10, 5)),
    ],
)
def test_sampler_from_bounds(npoints, expected_shape, random_bounds):
    sampler = SpaceSampler()
    xbounds, ybounds = random_bounds
    grid = sampler(bounds=[*xbounds, *ybounds], npoints=npoints)
    assert len(grid) == 2
    assert all(tuple(_.shape == expected_shape for _ in grid))
    np.testing.assert_array_equal([grid[0].min(), grid[0].max()], xbounds)
    np.testing.assert_array_equal([grid[1].min(), grid[1].max()], ybounds)


@pytest.mark.parametrize(
    "npoints,expected_shape",
    [
        (5, (5, 5)),
        ((10, 5), (5, 10)),
        ((5, 10), (10, 5)),
    ],
)
def test_sampler_from_mpl_axis(npoints, expected_shape, random_bounds):
    fig, ax = plt.subplots()
    sampler = SpaceSampler()
    xbounds, ybounds = random_bounds
    ax.set(xlim=xbounds, ylim=ybounds)
    grid = sampler(ax=ax, npoints=npoints)
    assert len(grid) == 2
    assert all(_.shape == expected_shape for _ in grid)
    np.testing.assert_array_equal([grid[0].min(), grid[0].max()], xbounds)
    np.testing.assert_array_equal([grid[1].min(), grid[1].max()], ybounds)
