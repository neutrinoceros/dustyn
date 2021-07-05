import abc
import sys
import warnings
from datetime import datetime
from time import ctime
from typing import Callable, Optional

import numpy as np

from dustyn._typing import FloatLike
from dustyn.core.record import Record

from .evolve import EvolutionModel
from .progress import DummyProgressBar, get_pbar


class SolverRecord(Record):
    def __init__(self, *args, **kwargs):

        print(
            "warning: the SolverRecord class is now identical to Record (except for this warning).",
            file=sys.stderr,
        )
        super().__init__(*args, **kwargs)


class Solver(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def step(
        R: np.ndarray, time: FloatLike, timestep: FloatLike, func: Callable
    ) -> None:
        """This should walk the R vector state over one timestep according to the evolution
        function `func` with signature

        def func(time, R):
            ...
        """
        pass

    @staticmethod
    def _update_from_buffer(output: list[np.ndarray], buffer=np.ndarray) -> None:
        output.append(buffer.copy())

    @classmethod
    def integrate(
        cls,
        model: EvolutionModel,
        R0: np.ndarray,
        *,
        tstart: FloatLike,
        tstop: FloatLike,
        CFL: FloatLike,
        pbar: bool = False,
        buffer_size: int = 1000,
        dt_record: Optional[FloatLike] = None,
        dt_min: FloatLike = 0.0,
        label: Optional[str] = None,
        logging: bool = False,
    ) -> Record:

        if CFL > 1.0:
            warnings.warn(
                f"Received {CFL=} > 1, stability is not guaranteed and results will likely be inaccurate."
            )
        elif CFL < 0:
            raise ValueError("Received negative CFL security.")
        elif CFL < (min_float := np.finfo(R0.dtype).eps):
            raise ValueError(
                f"Received too low value for {CFL=} . The minimal value is {min_float}"
            )

        tstart, tstop, CFL = map(R0.dtype.type, (tstart, tstop, CFL))

        if pbar:
            if buffer_size < 1000:
                warnings.warn(
                    "refreshing the progress bar too often is extremely expensive. "
                    "It is strongly advised to use progress bars only with `buffer_size` >= 1000."
                )
            if (
                dt_record is not None
                and int((tstop - tstart) // dt_record) <= 2 * buffer_size
            ):
                warnings.warn(
                    "Time resolution is too low to allow useful granulosity in the progressbar."
                )
            if logging:
                warnings.warn(
                    "logging and pbar can not be used simultaneously. Turning logging off."
                )
                logging = False

            title = label if label is not None else "Integrating..."
            progress = get_pbar(title=title, total=tstop)
        else:
            progress = DummyProgressBar()

        metadata = dict(
            CFL=CFL,
            label=label,
            geometry=getattr(model, "geometry", None),
            dimensionality=getattr(model, "dimensionality", None),
            stop_reason="completed",
        )
        states_buffer = np.empty((buffer_size, *R0.shape), dtype=R0.dtype)
        times_buffer = np.empty(buffer_size, dtype=R0.dtype)

        buffer_pos = 0
        R = R0.copy()
        t = tstart

        times: list[np.ndarray] = [np.array([tstart])]
        states: list[np.ndarray] = [np.expand_dims(R0, axis=0)]
        rec_count = 1
        next_trec = dt_record
        timestep_phy = CFL * model.get_max_timestep(R)
        it = 0
        while t < tstop:
            if dt_record is None:
                timestep_rec = min(timestep_phy, tstop - t)
                next_trec = t + timestep_rec
            else:
                timestep_rec = next_trec - t
            to_rec: bool = timestep_rec <= timestep_phy
            timestep = timestep_rec if to_rec else timestep_phy

            cls.step(R, time=t, timestep=timestep, func=model.evolve)

            t += timestep

            # this is essentially equivalent to
            # >>> timestep = min(CFL * model.get_max_timestep(R), tsave - t)
            # but with the advantage of being floating-point comparison proof
            # when we need to decide wether or not to save later.
            timestep_phy = CFL * model.get_max_timestep(R)
            nan_timestep: bool = timestep_phy != timestep_phy

            if to_rec:
                # t is updated manually here to secure floating point
                # equality check in the buffer updating block
                np.testing.assert_approx_equal(t, next_trec, significant=14)
                t = next_trec
                states_buffer[buffer_pos] = R
                times_buffer[buffer_pos] = t
                buffer_pos += 1
                rec_count += 1
                if dt_record is not None:
                    next_trec = min(dt_record * rec_count, tstop)

                if logging:
                    print(
                        f"{ctime(datetime.now().timestamp())} | {t=} | {it=}",
                        file=sys.stderr,
                    )
            elif nan_timestep or timestep < dt_min:
                if logging:
                    warnings.warn(
                        f"Aborting run with {timestep=} (minvalue is {dt_min=})"
                    )
                states_buffer[buffer_pos] = R
                times_buffer[buffer_pos] = t
                buffer_pos += 1
                rec_count += 1
                if logging:
                    print(
                        f"{ctime(datetime.now().timestamp())} | {t=} | {it=}",
                        file=sys.stderr,
                    )

            if (
                buffer_pos == buffer_size
                or t == tstop
                or nan_timestep
                or timestep < dt_min
            ):
                cls._update_from_buffer(states, states_buffer[:buffer_pos])
                cls._update_from_buffer(times, times_buffer[:buffer_pos])
                progress.update(completed=t, refresh=True)
                buffer_pos = 0

            it += 1

            if nan_timestep:
                metadata["stop_reason"] = "NaN timestep"
                break
            elif timestep < dt_min:
                metadata["stop_reason"] = "reached minimal timestep"
                break

        progress.update(completed=tstop, refresh=True)
        progress.stop()

        return Record(
            times=np.concatenate(times),
            states=np.concatenate(states),
            metadata=metadata,
        )


class RK4(Solver):
    """
    This is the Runge-Kutta method, with order 4
    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """

    @staticmethod
    def step(
        R: np.ndarray, time: FloatLike, timestep: FloatLike, func: Callable
    ) -> None:
        # - I'm avoiding premature optimisation, so I'm keeping the details of time positioning (t+dt)
        #   even though it should make no difference in my applications
        k1 = func(time, R)
        k2 = func(time + timestep / 2, R + k1 * timestep / 2)
        k3 = func(time + timestep / 2, R + k2 * timestep / 2)
        k4 = func(time + timestep, R + k3 * timestep)
        R += timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
