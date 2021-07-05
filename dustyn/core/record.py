import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dustyn._typing import PathLike


class Record:
    def __init__(
        self,
        times: np.ndarray,
        states: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        dirname: Optional[str] = None,
    ):
        self.times = times
        self.states = states
        self.metadata = metadata
        self.dirname = dirname

    def __len__(self):
        return len(self.times)

    @property
    def shape(self):
        return self.states.shape

    def save(
        self, dirname: PathLike, force: bool = False, extra: Optional[dict] = None
    ) -> None:
        dirpath = Path(dirname)
        if not dirpath.is_dir():
            os.makedirs(dirpath)
        elif not force:
            raise FileExistsError(dirname)

        with open(dirpath / "metadata.json", "w") as fh:
            json.dump(self.metadata, fh, indent=2)
            fh.write("\n")

        fields = {"times": self.times, "states": self.states}
        if extra is not None:
            fields.update(extra)
        for name, field in fields.items():
            np.save(dirpath / f"{name}.npy", field)

    @classmethod
    def load(cls, dirname: PathLike, full=False) -> "Record":
        with open(Path(dirname, "metadata.json")) as fh:
            metadata = json.load(fh)

        times = np.load(Path(dirname, "times.npy"))
        states = np.load(Path(dirname, "states.npy"))

        rec = cls(times=times, states=states, metadata=metadata, dirname=str(dirname))
        if full:
            rec.load_extra()
        return rec

    def load_extra(self) -> None:
        if self.dirname is None:
            raise ValueError(
                "This Record instance wasn't loaded from a directory, hence, no extra field is available."
            )
        for file in Path(self.dirname).glob("*.npy"):
            field_name = str(file.stem)
            if hasattr(self, field_name):
                continue
            setattr(self, field_name, np.load(file))
