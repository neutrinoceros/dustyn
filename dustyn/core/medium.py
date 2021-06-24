"""
A medium object represents the state of the fluid in which particles evolve.
"""


import abc
from typing import Callable, Mapping, Sequence

import numpy as np
import sympy as sp


class Medium(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get(self, field: str, *args) -> np.ndarray:
        pass


class AnalyticMedium(Medium):
    def __init__(self, symbols: Sequence[sp.Symbol], defs: Mapping[str, sp.Expr]):
        self._lambdas = {
            name: AnalyticMedium.lambdify(symbols, expr) for name, expr in defs.items()
        }

    def get(self, field: str, *args) -> np.ndarray:
        if field not in self._lambdas:
            raise KeyError(
                f"Unknown value `{field=}`. Known fields are {list(self._lambdas.keys())}."
            )
        return self._lambdas[field](*args)

    @staticmethod
    def lambdify(symbols: Sequence[sp.Symbol], expression: sp.Expr) -> Callable:
        """This is specialized replacement for `sympy.lambdify`
        that produce shape-preserving numpy lambdas, with `sympy.lambdify`
        returns scalars when passed constant expressions.

        See https://github.com/sympy/sympy/issues/5642
        """
        f = sp.lambdify(symbols, expression, "numpy")
        if isinstance(expression, (int, float, complex, np.number)):
            return np.vectorize(f)
        return f
