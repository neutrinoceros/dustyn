import json
import os
import warnings
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np

from dustyn._typing import PathLike


def get_suffix(i: int):
    if i == 0:
        return ""
    else:
        return f"_{i}"


def get_metadata_path(dirpath, i: int):
    return dirpath / f"metadata{get_suffix(i)}.json"


class Record:
    def __init__(
        self,
        times: np.ndarray,
        states: np.ndarray,
        *,
        timesteps: Optional[np.ndarray] = None,
        metadata: Optional[dict[str, Any]] = None,
        dirname: Optional[str] = None,
    ):
        self.times = times
        self.states = states
        self.timesteps = timesteps
        self.metadata = metadata
        self.dirname = dirname

    def __len__(self):
        return len(self.times)

    @property
    def shape(self):
        return self.states.shape

    def save(
        self,
        dirname: PathLike,
        *,
        extra: Optional[dict] = None,
        mode: str = "create",
        force: Optional[bool] = None,
    ) -> None:
        if force is not None:
            warnings.warn(
                "the 'force' keyword is deprecated, use mode='overwrite' instead.",
                UserWarning,
            )
            mode = "overwrite"
        if mode not in (valid_modes := ("create", "append", "overwrite")):
            raise ValueError(f"Got {mode!r}, expected one of {valid_modes}")

        dirpath = Path(dirname)
        if mode == "create" and dirpath.is_dir():
            raise FileExistsError(
                f"{str(dirname)!r} already exists. "
                "Use mode='append' to add a new entry "
                "or mode='overwrite' to erase the previous record."
            )

        os.makedirs(dirpath, exist_ok=True)

        save_num = 0
        while (metadata_path := get_metadata_path(dirpath, save_num)).exists():
            save_num += 1

        if save_num > 1:
            if mode == "create":
                raise FileExistsError(metadata_path)
            if mode == "overwrite":
                raise RuntimeError(
                    "Cannot overwrite a data dir with more than one record"
                )
        if mode == "overwrite":
            save_num = 0
            metadata_path = get_metadata_path(dirpath, save_num)

        with open(metadata_path, "w") as fh:
            json.dump(self.metadata, fh, indent=2)
            fh.write("\n")

        fields = {
            "times": self.times,
            "states": self.states,
            "timesteps": self.timesteps,
        }
        if extra is not None:
            fields.update(extra)
        for name, field in fields.items():
            if field is None:
                # don't save optional attributes
                continue
            np.save(dirpath / f"{name}{get_suffix(save_num)}.npy", field)

    @classmethod
    def load(
        cls, dirname: PathLike, *, extra: bool = False, full: Optional[bool] = None
    ) -> "Record":
        if full is not None:
            warnings.warn(
                "the 'full' keyword is deprecated, use 'extra' instead.", UserWarning
            )
            extra = full

        dirpath = Path(dirname)
        if not dirpath.is_dir():
            raise FileNotFoundError(f"No such file or directory {str(dirpath)!r}")

        save_num = 0
        if not (metadata_path := get_metadata_path(dirpath, save_num)).exists():
            raise FileNotFoundError(f"No such file or directory {str(metadata_path)!r}")

        metadata = {}
        _times = []
        _dts = []
        _states = []
        while (metadata_path := get_metadata_path(dirpath, save_num)).exists():
            suffix = get_suffix(save_num)
            with open(metadata_path) as fh:
                metadata.update(json.load(fh))

            _times.append(np.load(dirpath / f"times{suffix}.npy"))
            try:
                _dts.append(np.load(dirpath / f"timesteps{suffix}.npy"))
            except FileNotFoundError:
                _dts.append(np.full_like(_times[-1], np.nan))
            _states.append(np.load(dirpath / f"states{suffix}.npy"))

            save_num += 1

        rec = cls(
            times=np.concatenate(_times),
            states=np.concatenate(_states),
            timesteps=np.concatenate(_dts),
            metadata=metadata,
            dirname=str(dirname),
        )
        if extra:
            rec.load_extra()
        return rec

    def load_extra(self) -> None:
        if self.dirname is None:
            raise ValueError(
                "This Record instance wasn't loaded from a directory, "
                "hence, no extra field is available."
            )

        field_names = set()
        for file in Path(self.dirname).glob("*.npy"):
            name, *_suffix = file.stem.rsplit("_")
            field_names.add(name)

        for name in field_names:
            files = sorted(Path(self.dirname).glob(f"{name}*.npy"))
            data = np.concatenate([np.load(_) for _ in files])
            if hasattr(self, name):
                continue
            setattr(self, name, data)
