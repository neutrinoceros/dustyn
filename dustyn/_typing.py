import os
from typing import Iterable
from typing import TypeVar
from typing import Union

import numpy as np

# PEP 484 specify that integers are acceptable where floats are required
FloatLike = Union[float, np.floating]

# types that are supposed to support pathlib.Path (natively, str and classes implementing the __fspath__ protocol)
PathLike = Union[str, os.PathLike[str]]

T = TypeVar("T")

SingleOrDouble = Union[T, tuple[T, T]]
SingleOrMultiple = Union[T, Iterable[T]]
