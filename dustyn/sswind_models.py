"""
This module defines specialized classes to read self similar 1D simulation
datasets from Lesur 2021, available at https://github.com/glesur/PPDwind
"""

import re
import warnings
from functools import cached_property
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import interp1d

from dustyn._typing import FloatLike, PathLike
from dustyn.core.medium import Medium
from dustyn.disk import SphericalDisk3D


class SSWDataset:
    HEADER_SIZE = 7

    def __init__(self, fname: PathLike) -> None:
        columns = self.get_colunms(fname)
        self._data = self.load_as_ndarray(fname, columns)
        self._fields = columns

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def get_colunms(self, fname: PathLike) -> list[str]:
        with open(fname) as fh:
            for _ in range(self.HEADER_SIZE - 1):
                next(fh)
            header = fh.readline()
        return re.sub(r"\([\w\s]+\)\s", " ", header).split()

    def load_as_ndarray(
        self, fname: PathLike, columns: Optional[list[str]] = None
    ) -> np.ndarray:
        if columns is None:
            columns = self.get_colunms(fname)
        return np.loadtxt(
            fname,
            skiprows=self.HEADER_SIZE,
            dtype=[(name, "float64") for name in columns],
        )


class SSWMedium(Medium):
    """
    Self-similarity (SS) is central here.
    SS (+ axisymmetry) is equivalent to assuming that any function of space is separable as

    f(r, theta) = f1(r) x f2(theta)
    with
    f1(r) = r^zeta
    where zeta is a constant. Note that, as discussed in the paper, this description
    is valid with either r representing the spherical OR cylindrical radius, though the value
    of zeta and the f2 functions require appropriate transformations to convert between
    coordinate systems.

    In this implementation, f1 is expressed via the `get_zeta` method,
    while functions f2 are stored as 1D anonymous function in `self._lambdas`
    """

    # This is a constant in the whole paper, see section 3.1
    aspect_ratio = 0.1

    self_similar_scaling_exponents: dict[str, float] = {
        # density
        "rho": -1.5,
        # thermal pressure
        "P": -2.5,
        # velocity
        "v": -0.5,
        # magnetic field
        "B": -1.25,
        # ambipolar number
        "Lambda_A": 0.0,
        # Ohmic number
        "Rm": 0.0,
    }

    field_aliases: dict[str, str] = {
        "density": "rho",
        "velocity_r": "vr",
        "velocity_theta": "vtheta",
        "velocity_phi": "vphi",
    }

    def __init__(self, fname: PathLike):
        self.ds = SSWDataset(fname)

        self._lambdas: dict[str, Callable] = {
            field: interp1d(self.ds["theta"], self.ds[field])
            for field in self.ds._fields
        }

        # models are assumed locally isothermal, i.e., sound speed
        # is a pure function of the cylindrical radius R
        # In other words, we need to return the corresponding midplane value, which reads
        # cs_mid = H(R) * Omega_K(R)
        #        = h * R * sqrt(GM/R^3)
        #        = h * r * sin(theta) * sqrt(GM/(r^3 sin^3(theta)))
        #        = r^(-1/2) * h * sqrt(GM/sin(theta))
        # where R is the cylindrical radius here, while r is the spherical radius,
        # and h is the aspect ratio (which is constant in our application)
        # NOTE: I'm assuming the raw data is always using GM=1
        self._lambdas["sound_speed"] = (
            lambda theta: self.aspect_ratio * np.sin(theta) ** -0.5
        )

    @cached_property
    def field_list(self) -> list[str]:
        return list(self._lambdas.keys())

    def get(self, field: str, *args) -> np.ndarray:

        # simulations are actually are axisymetric so there the phi component is not used
        # with PEP 622 we could more easily support flexible inputs as
        # match args:
        #     case r, theta, _phi:
        #         pass
        #     case r, theta:
        #         pass
        #     case _:
        #         raise TypeError("At least two arguments expected.")
        r, theta, phi = args

        field = self.__class__.field_aliases.get(field, field)

        if field not in self._lambdas:
            raise KeyError(
                f"Unknown value `{field=}`. Known fields are {self.field_list}."
            )

        try:
            rv = self._lambdas[field](theta)
        except ValueError:
            msg = "Invalid values encountered. "
            if (min_eff := theta.min()) < (min_valid := self.ds["theta"].min()):
                msg += f"Min possible theta is {min_valid} while input has min(theta) = {min_eff} "
            if (max_eff := theta.max()) > (max_valid := self.ds["theta"].max()):
                msg += f"Max possible theta is {max_valid} while input has max(theta) = {max_eff}"
            warnings.warn(msg)
            if hasattr(r, "shape"):
                rv = np.full_like(r, np.nan)
            else:
                # assume scalar input
                rv = np.nan

        return r ** self._get_zeta(field) * rv

    def _get_zeta(self, field: str) -> float:
        """Retrieve the appropriate scaling factor"""
        if field == "sound_speed":
            return self.self_similar_scaling_exponents["v"]
        for prefix, zeta in self.self_similar_scaling_exponents.items():
            if field.startswith(prefix):
                return zeta
        raise KeyError(f"Could not find appropriate scaling factor for {field=}")


class SSWDisk(SphericalDisk3D):
    def __init__(self, mass_to_surface: FloatLike, **kwargs):
        # I define the mass to surface ratio as the product of particle radius
        # and particle intrinsic density, which, akin to "GM" in equations of
        # motion for test particles, only ever appear as a product.
        self.mass_to_surface = mass_to_surface
        super().__init__(**kwargs)

    def get_stopping_time(self, R: np.ndarray) -> FloatLike:
        # This assumes the Epstein regime is relevant
        # TODO: check implementation (a dimensionless geometric factor may be missing)
        return self.mass_to_surface / (
            self.medium.get("density", *R[3:]) * self.medium.get("sound_speed", *R[3:])
        )
