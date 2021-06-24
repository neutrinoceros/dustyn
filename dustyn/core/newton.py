import abc
import warnings
from typing import Union

import numpy as np

from dustyn._typing import FloatLike

from .algebra import cross_product
from .evolve import EvolutionModel
from .transform import spherical2cartesian


class Newton(EvolutionModel, abc.ABC):
    dimensionality: int
    geometry: str

    def __init__(self, GM: FloatLike = 1.0):
        self.GM = GM

    @abc.abstractmethod
    def hamiltonian(self, R: np.ndarray) -> Union[FloatLike, np.ndarray]:
        """Compute the hamiltonian of a given state.
        This quantity is supposed to be conserved through the integration as is
        useful for testing.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def spherical_radius_squared(cls, R: np.ndarray) -> FloatLike:
        pass

    def get_max_timestep(
        self,
        R: np.ndarray,
    ) -> FloatLike:
        # the relevant maximal timestep dictated by the gravitational force reads
        # dt = 1 / Omega_K
        #    = sqrt(r^3 / GM)
        #    = ((r^2)^3/2)^1/2 * (GM)^-1/2
        #    = r2^3/4 * (GM)^-1/2
        return self.__class__.spherical_radius_squared(R) ** 0.75 * self.GM ** (-0.5)


class NewtonCartesian(Newton, abc.ABC):
    geometry = "cartesian"

    @classmethod
    def spherical_radius_squared(cls, R: np.ndarray) -> FloatLike:
        return np.sum(R[cls.dimensionality :] ** 2, axis=0)


class NewtonCartesian2D(NewtonCartesian):
    dimensionality = 2

    def evolve(self, t: float, R: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the time evolution of a state vector R
        such that R = [vx, vy, x, y]
        """
        vx, vy, x, y = R
        phi = np.arctan2(y, x)
        r2 = self.__class__.spherical_radius_squared(R)

        tot_force = -self.GM / r2
        return [tot_force * np.cos(phi), tot_force * np.sin(phi), vx, vy]

    def hamiltonian(self, R: np.ndarray) -> Union[float, np.ndarray]:
        vx, vy, x, y = R
        r2 = self.__class__.spherical_radius_squared(R)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"divide by zero encountered in true_divide"
            )
            return 0.5 * (vx ** 2 + vy ** 2) - self.GM / np.sqrt(r2)


class NewtonCartesian3D(NewtonCartesian):
    dimensionality = 3

    def evolve(self, t: float, R: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the time evolution of a state vector R
        such that R = [vx, vy, vz, x, y, z]
        """
        vx, vy, vz, x, y, z = R
        phi = np.arctan2(y, x)

        rho = np.sqrt(x ** 2 + y ** 2)  # cylindrical radius
        theta = np.arctan2(rho, z)

        r2 = self.__class__.spherical_radius_squared(R)

        tot_force = -self.GM / r2
        return [
            tot_force * np.cos(phi) * np.sin(theta),
            tot_force * np.sin(phi) * np.sin(theta),
            tot_force * np.cos(theta),
            vx,
            vy,
            vz,
        ]

    def hamiltonian(self, R: np.ndarray) -> Union[float, np.ndarray]:
        vx, vy, vz, x, y, z = R

        r2 = self.__class__.spherical_radius_squared(R)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"divide by zero encountered in true_divide"
            )
            return 0.5 * (vx ** 2 + vy ** 2 + vz ** 2) - self.GM / np.sqrt(r2)


class NewtonCurvilinear(Newton, abc.ABC):
    # regroup polar, spherical and cylindrical geometries
    @classmethod
    def spherical_radius_squared(cls, R: np.ndarray) -> FloatLike:
        return R[cls.dimensionality] ** 2


class NewtonPolar2D(NewtonCurvilinear):
    """this is an exercise before getting to full 3D spherical
    The coordinate system is de-facto polar (r, phi).
    """

    dimensionality = 2
    geometry = "cylindrical"

    def evolve(self, t: float, R: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the time evolution of a state vector R
        such that R = [vr, vphi, r, phi]
        """
        vr, vphi, r, phi = R

        g = -self.GM / r ** 2
        return [
            g + vphi ** 2 / r,
            -vr * vphi / r,
            vr,
            vphi / r,
        ]

    def hamiltonian(self, R: np.ndarray) -> Union[float, np.ndarray]:
        vr, vphi, r, phi = R

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"divide by zero encountered in true_divide"
            )
            return 0.5 * (vr ** 2 + vphi ** 2) - self.GM / r


class NewtonSpherical3D(NewtonCurvilinear):
    dimensionality = 3
    geometry = "spherical"

    def evolve(self, t: float, R: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the time evolution of a state vector R
        such that R = [vr, vtheta, vphi, r, theta, phi]
        """
        vr, vtheta, vphi, r, theta, phi = R

        g = -self.GM / r ** 2
        return np.array(
            [
                g + (vphi ** 2 + vtheta ** 2) / r,
                (vphi ** 2 / np.tan(theta) - vr * vtheta) / r,
                -vphi * (vr + vtheta / np.tan(theta)) / r,
                vr,
                vtheta / r,
                vphi / (r * np.sin(theta)),
            ]
        )

    def hamiltonian(self, R: np.ndarray) -> Union[float, np.ndarray]:
        vr, vtheta, vphi, r, theta, phi = R

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"divide by zero encountered in true_divide"
            )
            return 0.5 * (vr ** 2 + vphi ** 2 + vtheta ** 2) - self.GM / r

    def angular_momentum(self, R: np.ndarray) -> np.ndarray:
        """Compute the CARTESIAN components of the angular momentum."""
        if R.ndim > 2:
            raise NotImplementedError(
                "This may work but I haven't tried it yet. "
                "Remove this `raise` statement if you have faith."
            )

        ncols = 1 if R.ndim == 1 else R.shape[1]
        r = np.zeros((3, ncols), dtype="float64").squeeze()
        r[0] = R[3]
        v = np.array(R[:3])
        L = cross_product(r, v)
        return spherical2cartesian(position=R[3:7], vector=L)
