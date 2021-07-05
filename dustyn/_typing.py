import os
from typing import Iterable, TypeVar, Union

import numpy as np

FloatLike = Union[float, np.floating, int, np.integer]
PathLike = Union[str, os.PathLike[str]]

T = TypeVar("T")

SingleOrDouble = Union[T, tuple[T, T]]
SingleOrMultiple = Union[T, Iterable[T]]
