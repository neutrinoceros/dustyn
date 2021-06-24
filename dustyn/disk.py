import abc

import numpy as np

from dustyn._typing import FloatLike
from dustyn.core.medium import Medium
from dustyn.core.newton import Newton, NewtonSpherical3D


class Disk(Newton, abc.ABC):
    def __init__(
        self, medium: Medium, short_friction_security: FloatLike = 0.0, **kwargs
    ):
        """
        short_friction_security is disabled by default.
        Use a real number between 0 (excluded) and 1 (included) to activate it.
        """
        if short_friction_security > 1:
            raise ValueError
        self.medium = medium
        self._sf_security = short_friction_security
        self._short_friction = False
        super().__init__(**kwargs)

    @abc.abstractmethod
    def _evolve_short_friction(self, t: FloatLike, R: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_stopping_time(self, R: np.ndarray) -> FloatLike:
        pass

    @abc.abstractmethod
    def get_drag(self, R: np.ndarray):
        pass

    def evolve(self, t: float, R: np.ndarray, **kwargs) -> np.ndarray:
        if self._short_friction:
            # because the following algo involves geometric terms
            # it is deleguated to child classes where the geometry is specified
            return self._evolve_short_friction(t, R)
        ret = super().evolve(t, R, **kwargs)
        ret[: self.dimensionality] += self.get_drag(R)
        return ret

    @abc.abstractmethod
    def get_gas_velocity(self, position: np.ndarray) -> np.ndarray:
        pass

    def get_max_timestep(
        self,
        R: np.ndarray,
    ) -> FloatLike:
        dt_kep = super().get_max_timestep(R)
        dt_drag = 2 * self.get_stopping_time(R)

        min_dt = self._sf_security * dt_kep
        self._short_friction = min_dt > dt_drag
        if self._short_friction:
            return min_dt
        return min(dt_kep, dt_drag)


class SphericalDisk3D(Disk, NewtonSpherical3D, abc.ABC):
    def get_gas_velocity(self, position: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.medium.get(f"velocity_{k}", *position)
                for k in ("r", "theta", "phi")
            ],
            dtype="float64",
        )

    def get_drag(self, R: np.ndarray):
        vp = R[: self.dimensionality]
        vg = self.get_gas_velocity(R[self.dimensionality :])
        delta_v = vg - vp
        return delta_v / self.get_stopping_time(R)

    def _evolve_short_friction(self, t: FloatLike, R: np.ndarray) -> np.ndarray:
        # this method is special in that is will update the state vector R itself,
        # which isn't normally done in other `evolve` methods

        # get acceleration from gravity and geometry
        gacc = NewtonSpherical3D.evolve(self, t, R)[: self.dimensionality]

        ret = np.zeros_like(R)
        # We don't evolve v, we set it by hand (nasty trick, granted)
        # force v to be what we want
        vr, vtheta, vphi = R[: self.dimensionality] = (
            self.get_gas_velocity(R[self.dimensionality :])
            + self.get_stopping_time(R) * gacc
        )

        # update positions following the current velocity
        # this is exactly the same as what is done in NewtonSpherical3D.evolve
        # maybe there's a smart way to avoid code duplication
        r, theta, phi = R[self.dimensionality :]
        ret[self.dimensionality :] = [vr, vtheta / r, vphi / (r * np.sin(theta))]
        return ret


# and now for some concrete implementations
class ConstantTauSphereDisk(SphericalDisk3D):
    def __init__(self, stopping_time: FloatLike = 1.0, **kwargs):
        self.stopping_time = stopping_time
        super().__init__(**kwargs)

    def get_stopping_time(self, R: np.ndarray) -> FloatLike:
        return self.stopping_time


class ConstantStokesSphereDisk(SphericalDisk3D):
    def __init__(self, stokes: FloatLike = 1.0, **kwargs):
        self.stokes = stokes
        super().__init__(**kwargs)

    def get_stopping_time(self, R: np.ndarray) -> FloatLike:
        return (
            self.stokes * self.__class__.spherical_radius_squared(R) ** 0.75 * self.GM
        )
