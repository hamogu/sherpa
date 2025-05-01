#
#  Copyright (C) 2025
#  MIT
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
"""Interface to Scipy optimization methods.

This module contains classes that wrap the optimization functions in
`scipy.optimize` to match the calling signature and return values
to the Sherpa interface.

If `scipy <https://scipy.org>`_ is installed, classes are created automatically and
can be used in the same way as other optimizers in Sherpa.
The most versatile function is `scipy.optimize.minimize`, wrapped into the
`Scipy_Minimize` class.
`scipy.optimize.minimize` is itself a wrapper around several different
optimization algorithms. Which one is used by default depends on the bounds
places on the parameter values of the model to be fit.

`scipy.optimize` also contains several global optimizers that aim to explore
the parameter space more fully. Most of these will only work if meaningful
limits are placed in the parameters, see
`the scipy docs for global optimizers <https://docs.scipy.org/doc/scipy/tutorial/optimize.html#global-optimization>`_
for details.
"""
from collections.abc import Callable
import functools
import inspect
import re
from textwrap import dedent
from typing import Any


import numpy as np
from sherpa.optmethods.buildin import OptMethod
from sherpa.utils.types import (ArrayType,
                                OptFunc, OptReturn,
                                StatFunc)
from sherpa.models.parameter import hugeval

# Will be filled out programmatically at the bottom of this file
__all__ = []


re_next_param_line = re.compile('^[a-zA-Z]')
re_sphinx_ref = [re.compile(":[\w-]+:`[ \w<>()-_.]+?`"),
 re.compile(".. versionadded:: [\d.]+"),
 re.compile(".. versionchanged:: [\d.]+"),
 re.compile(".. deprecated:: [\d.]+"),
]

# numpydoc.docscrape.NumpyDocString offers an easier interface, but
# we can extract this simple information without that dependency.
def extract_parameter_text_from_docstring(doc: str, par: str, cleanup=True) -> str:
    """Extract the description for a parameter from a docstring in numpy format.

    Parameters
    ----------
    doc : str
        The docstring to extract the parameter text from.
    par : str
        The name of the parameter to extract the text for.
    cleanup : bool
        If True, remove Sphinx references from the text.

    Returns
    -------
    str
        The text for the parameter.

    Notes
    -----
    ``numpydoc.docscrape.NumpyDocString`` offers an easier interface,
    but this function achieves the limited goal without that dependency.
    """
    # For some functions in scipy the entire docstring is indented except
    # for the first line.
    doc = dedent(doc.split('\n', 1)[1])

    out = []
    good = False
    for l in doc.split('\n'):
        if good and re_next_param_line.match(l):
            break
        if l.startswith(f'{par}'):
            good = True
        if good:
            if cleanup:
                for r in re_sphinx_ref:
                    l = r.sub("", l)
            out.append(l)
    outdoc = dedent('\n'.join(out)).rstrip('\n')
    if not outdoc:
        raise ValueError(f"Could not find parameter {par} in docstring")
    return outdoc

def convert_bounds_to_scipy(parmins: ArrayType,
                            parmaxes: ArrayType,
                            requires_finite_bounds: bool) -> list[tuple[float | None, float | None]] | None:
    """Convert parmins and parmaxes to format used by `scipy.optimize` for bounds.

    Parameters
    ----------
    parmins : ArrayType
        The minimum values for the parameters.
    parmaxes : ArrayType
        The maximum values for the parameters.
    requires_finite_bounds : bool
        If True, the scipy function requires finite bounds.

    Returns
    -------
    list[tuple[float | None, float | None]] | None
        The bounds in the format used by scipy.optimize, or None if
        there are no bounds.
    """
    if np.allclose(parmins, -hugeval) and np.allclose(parmaxes, hugeval):
        bounds = None
    else:
        bounds = [(None if pmin == -hugeval else pmin,
                 None if pmax == hugeval else pmax)
                for pmin, pmax in zip(parmins, parmaxes)]
    if requires_finite_bounds:
        if None in np.array(bounds):
            raise ValueError("The scipy function requires finite bounds, but "
                             "the Sherpa model has some bounds set to HUGEVAL.")
    return bounds


def wrap_scipy_fcn(func: Callable,
                   requires_finite_bounds: bool) -> OptFunc:
    """Wrap a function in scipy.optimize to the Sherpa interface.
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def fcn(fcn: StatFunc,
         x0: np.ndarray,
         xmin: np.ndarray,
         xmax: np.ndarray,
         **kwargs) -> OptReturn:

        def fcn_wrapper(x):
            # The function is called with the parameters
            # and returns the statistic and per-bin values
            return fcn(x)[0]

        converted_args: dict = {}
        if 'x0' in sig.parameters.keys():
            converted_args['x0'] = x0
        if 'bounds' in sig.parameters.keys():
            converted_args['bounds'] = convert_bounds_to_scipy(xmin, xmax,
                                                                requires_finite_bounds)

        result = func(fcn_wrapper, **converted_args, **kwargs)
        for arg in ['bounds', 'ranges']:
            if arg in converted_args:
                result[f'input_{arg}'] = converted_args[arg]
        return (result.success, result.x, result.fun, result.message, result)
    return fcn


SCIPY_KEYWORDS_NOT_APPLICABLE = ['fun', 'func',
                                 'args', 'x0', 'bounds', 'ranges',
                                 'jac', 'hess', 'hessp',
                                 'constraints']
'''List of keywords NOT to expose when wrapping scipy optimization functions.

This is a list of keywords in the signature of functions in scipy.optimize that we do not
want to expose, either because the interface converts Sherpa input to them
automatically, or because they are not applicable to the Sherpa interface.
'''

_DOCSTRING_TEMPLATE = """Interface to scipy.optimize.{cls.scipy_func.__name__}

This class wraps the scipy optimizer {cls.scipy_func.__name__} to match the calling
signature and return values to the Sherpa interface.

Some of the keywords in the signature of the scipy function are not
applicable to the Sherpa interface or are prefilled by Sherpa.
The following arguments are exposed as attributes of this class and
can be changed. A short summary is below, but see the
`scipy.optimize.{cls.scipy_func.__name__}` documentation for details.

(The following text extracted from the docstring of the scipy function to ensure that it
matches the current scipy version, but it may contain unresolved references
and other formatting issues.)

Attributes
----------
"""

class ScipyBase(OptMethod):
    """Base class for wrapping scipy optimization functions.
    This class wraps that function to match the calling signature and
    return values to the Sherpa interface.
    """

    scipy_func: Callable
    """Optimization function in scipy

    This class wraps that function to match the calling signature and
    return values to the Sherpa interface.
    """
    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__doc__ = _DOCSTRING_TEMPLATE.format(cls=cls)
        for p in cls._get_default_config(cls).keys():
            cls.__doc__ += f"{extract_parameter_text_from_docstring(cls.scipy_func.__doc__, p)}\n"

    def __init__(self, name : str | None = None) -> None:
        super().__init__(name=f'scipy.optimize.{self.scipy_func.__name__}' if name is None else name,
                         optfunc=wrap_scipy_fcn(self.scipy_func,
                                                self.requires_finite_bounds),
                        )

    def _get_default_config(self) -> dict[str, Any]:
        sig = inspect.signature(self.scipy_func)
        return {p.name: p.default for p in sig.parameters.values()
                if p.kind == p.POSITIONAL_OR_KEYWORD and
                 p.name not in SCIPY_KEYWORDS_NOT_APPLICABLE}

    default_config = property(_get_default_config,
                              doc='The default settings for the optimiser.')


try:
    from scipy import optimize
    for func, requires_finite_bounds in [
        (optimize.minimize, False),
        (optimize.basinhopping, False),

        # Brute has a different interface and is almost
        # identical to Sherpa's GridSearch so there is little
        # value in wrapping it here.
        # optimize.brute,

        (optimize.differential_evolution, True),
        (optimize.shgo, True),
        (optimize.dual_annealing, True),
        (optimize.direct, True),
                 ]:
        myscipy = type(f'Scipy_{func.__name__.capitalize()}',
                       (ScipyBase,),
                       {'scipy_func': staticmethod(func),
                        'requires_finite_bounds': requires_finite_bounds})
        globals()[myscipy.__name__] = myscipy
        __all__.append(myscipy.__name__)

except ImportError:
    # scipy is not available, so we cannot create the classes
    # that wrap the scipy functions
    pass
