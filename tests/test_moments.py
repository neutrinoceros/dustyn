from random import randint
from random import random

import numpy as np
import pytest
from hypothesis import assume  # useful to discard free particles
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import builds
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from scipy.integrate import solve_ivp

from dustyn.core.newton import NewtonCartesian2D
from dustyn.core.newton import NewtonCartesian3D
from dustyn.core.newton import NewtonPolar2D
from dustyn.core.newton import NewtonSpherical3D

# devnotes
# - I don't see how to parametrize these tests to avoid duplicated code
# - only testing positive values for all components of the IC vector is obviously bad
#   I wrote this as a simple way to avoid having particles randomly drawn at 0,0,0
#   The more appropriate solution could be to use `hypothesis.strategies.build`


def my_random_point(f: float):
    return f * random() * (-1) ** randint(0, 1)


@pytest.mark.xfail(
    reason="ill cases like R=[0, 0, 0...] are not excluded. They used to be guarded by a 'capture' mechanism internally."
)
@given(
    arrays(dtype="float64", shape=4, elements=builds(my_random_point, floats(0, 100)))
)
def test_hamiltonian_conservation_cartesian_2D(R0):
    model = NewtonCartesian2D()
    sol = solve_ivp(fun=model.evolve, t_span=(0, 10), y0=R0, method="Radau")
    h = model.hamiltonian(sol.y)
    expected = model.hamiltonian(R0)
    np.testing.assert_allclose(h[np.isfinite(h)], expected, rtol=1e-3)


@pytest.mark.xfail(
    reason="ill cases like R=[0, 0, 0...] are not excluded. They used to be guarded by a 'capture' mechanism internally."
)
@given(
    arrays(dtype="float64", shape=6, elements=builds(my_random_point, floats(0, 100)))
)
def test_hamiltonian_conservation_cartesian_3D(R0):
    model = NewtonCartesian3D()
    sol = solve_ivp(fun=model.evolve, t_span=(0, 10), y0=R0, method="Radau")
    h = model.hamiltonian(sol.y)
    expected = model.hamiltonian(R0)
    np.testing.assert_allclose(h[np.isfinite(h)], expected, rtol=1e-3)


rs = floats(1, 100)
thetas = floats(0.1, np.pi - 0.1)
phis = floats(-6 * np.pi, 6 * np.pi)  # this should cover every case
vel_components = floats(-1e-2, 1e-2)


ns2 = NewtonPolar2D()
ns3 = NewtonSpherical3D()


@composite
def initial_states(draw, dim=3):
    """A custom strategy to generate inital states for particles in spherical 3D"""

    # R0 is written in spherical (vr, vtheta, vphi, r, theta, phi)
    R0 = [
        draw(vel_components),
        draw(vel_components),
        0,
        draw(rs),
        draw(thetas),
        draw(phis),
    ]
    # init with keplerian velocity
    R0[2] = (-1) ** randint(0, 1) * np.sqrt(ns3.GM / R0[3])

    # the following is a bit clunky because I wrote it
    # in a way that works in Polar 2D and Spherical 3D
    if dim == 3:
        model = ns3
    elif dim == 2:
        R0.pop(1)  # vtheta
        R0.pop(-2)  # theta
        model = ns2
    else:
        raise NotImplementedError

    H0 = model.hamiltonian(R0)
    assume(H0 < 0)  # reject free particles for this test
    assume(np.isfinite(H0))  # also reject trapped particles
    return np.array(R0)


@given(initial_states(dim=2))
def test_hamiltonian_conservation_spherical_2D(R0):
    sol = solve_ivp(fun=ns2.evolve, t_span=(0, 10), y0=R0, method="Radau")
    h = ns2.hamiltonian(sol.y)
    expected = ns2.hamiltonian(R0)
    np.testing.assert_allclose(h[np.isfinite(h)], expected, rtol=1e-6)


@given(initial_states())
def test_conservation_spherical_3D(R0):
    sol = solve_ivp(fun=ns3.evolve, t_span=(0, 10), y0=R0, method="Radau")
    h = ns3.hamiltonian(sol.y)
    expected = ns3.hamiltonian(R0)
    np.testing.assert_allclose(h[np.isfinite(h)], expected, rtol=1e-3)

    # angular momentum (vector) should be conserved
    L0 = ns3.angular_momentum(R0)
    L = ns3.angular_momentum(sol.y)

    diff = L.T - L0
    np.testing.assert_array_almost_equal(diff, 0, 2)
