import os
from typing import Iterable, Tuple, TypeVar, Union

import numpy as np

FloatLike = Union[float, np.floating, int, np.integer]
PathLike = Union[str, os.PathLike[str]]

T = TypeVar("T")

SingleOrDouble = Union[T, Tuple[T, T]]
SingleOrMultiple = Union[T, Iterable[T]]
