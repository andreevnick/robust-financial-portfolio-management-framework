# Copyright 2021 portfolio-robustfpm-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

r""" This submodule implements :class:`MultivalMap` as an abstract interface for multivalued mappings and some specific mappings.

For any sets :math:`X, Y`, a multivalued mapping is such a mapping :math:`\Gamma: X \mapsto Y`, that

.. math :: \forall x \in X \;\; \Gamma(x) \subseteq Y

This module also provides two simple instances of :class:`IMultivalMap`: `NoConstraints` and `LongOnlyConstraints` for
ease of use. They are defined as:

>>> NoConstraints = IdenticalMap(RealSpaceHandler())
>>> LongOnlyConstraints = IdenticalMap(NonNegativeSpaceHandler())

See Also
--------
:class:`robustfpm.pricing.set_handler.RealSpaceHandler`
:class:`robustfpm.pricing.set_handler.NonNegativeSpaceHandler`
"""

import numpy as np
from abc import ABC, abstractmethod
from .set_handler import ISetHandler, NonNegativeSimplex, RealSpaceHandler, NonNegativeSpaceHandler
from .lattice import Lattice

__all__ = ['IMultivalMap',
           'IdenticalMap',
           'PriceDynamics',
           'PIDynamics',
           'ConstantDynamics',
           'MDAFDynamics',
           'SimplexConstraints',
           'NoConstraints',
           'LongOnlyConstraints'
           ]


class IMultivalMap(ABC):
    """ An abstract interface class for multivalued mappings"""

    @abstractmethod
    def __call__(self, x, t):
        r""" :code:`A.__call__(x,t)` is equivalent to :code:`A(x,t)`

        Parameters
        ----------
        x: np.ndarray
            A point or a set of points from :math:`\mathbb{R}^{n}`
        t: int
            Value of time

        Returns
        -------
        ISetHandler
            Value of a mapping at point(s) `x`, `t`
        """
        raise NotImplementedError('The method must be defined in a subclass')

    @property
    @abstractmethod
    def dim(self):
        r"""
            int: The dimension of image or :code:`np.inf` if return value can be of any dimension (e.g., a :class:`robustfpm.pricing.set_handler.RealSpaceHandler`)

        For a multivalued map :math:`\Gamma: \mathbb{R}^{n} \times  \{0, 1, \dots\} \mapsto \mathbb{R}^{n}`, returns :math:`n`.
        """
        raise NotImplementedError('The method must be defined in a subclass')


class IdenticalMap(IMultivalMap):
    """ Identical mapping: always returns the same set (its support)

    Parameters
    ----------
    support: ISetHandler
        The underlying set to return
    """

    def __init__(self, support: ISetHandler):
        self.support = support

    def __call__(self, x=0, t=0):
        return self.support

    @property
    def dim(self):
        return self.support.dim


class SimplexConstraints(IMultivalMap):
    r""" A simplex mapping that corresponds to the following trading constraints:

    - No short positions;
    - Total value of risky assets at any given time can not exceed given limit `r`

    In mathematical terms, given a vector of discounted prices of risky assets :math:`x = (x_1, \dots, x_n)`

    .. math:: D_t(x) = \{h = (h_1, \dots, h_n):\; \sum\limits_{i=1}^{n}h_i x_i \leqslant r, \; h_i \geqslant 0, i = 1,\dots, n\}

    Parameters
    ----------
    r : np.float64
        Limit on total value of risky assets (in discounted prices, i.e., in terms of units of riskless asset)

    Notes
    -----
    We denote these constraints as *simplex* since :math:`D_t(x)` is, in fact, a special type of N-simplex.

    See Also
    --------

    :class:`robustfpm.pricing.set_handler.NonNegativeSimplex`
    """

    def __init__(self, r: np.float64):
        if r < 0:
            raise ValueError('Bound must be non-negative!')
        self.r = r

    def __call__(self, x, t):
        return NonNegativeSimplex(self.r / x, dtype=x.dtype)

    @property
    def dim(self):
        return np.inf


class PriceDynamics(IMultivalMap):
    """An abstract base class for different price dynamics.
    Differs from :class:`IMultivalMap` in that it also has to have:

        - type
        - time horizon
        - method for calculating Lipschitz constant
    """
    _allowed_types = {'mult', 'add'}

    @abstractmethod
    def __call__(self, x, t=1):
        r""" :code:`A.__call__(x,t)` is equivalent to :code:`A(x,t)`

        Parameters
        ----------
        x: np.ndarray, size (`t`, n)
            A set of previous prices, each price — a point from :math:`\mathbb{R}^{n}`

        t: int
            Value of time

        Returns
        -------
        ISetHandler
            For additive dynamics, returns increment :math:`K_t(x)`.
            For multiplicative dynamics, returns multipliers :math:`C_t(x)`.
        """

        raise NotImplementedError('The method must be defined in a subclass')

    @property
    @abstractmethod
    def type(self):
        """
        {'mult', 'add'}: Type of price dynamics, either multiplicative or additive"""
        raise NotImplementedError('The method must be defined in a subclass')

    @property
    @abstractmethod
    def t_max(self):
        """int: Time horizon"""
        raise NotImplementedError('The method must be defined in a subclass')

    @abstractmethod
    def get_lipschitz(self, t: int):
        """ Get value of a Lipshitz constant for underlying multivalued mapping at time `t`.

        Parameters
        ----------
        t: int
            Time horizon

        Returns
        -------
        np.float64
            Lipschitz constant of :math:`K_t(x)`
        """
        raise NotImplementedError('The method must be defined in a subclass')


class PIDynamics(PriceDynamics):
    """ Abstract class for price-independent price dynamics.
    That is, increment (for additive) or multipliers (for multiplicative) is independent of x.
    """

    def __call__(self, x=0, t=1):
        return self._call(t)

    @abstractmethod
    def _call(self, t) -> ISetHandler:
        raise NotImplementedError('The method must be defined in a subclass')

    def get_lipschitz(self, t: int):
        if self.type == 'add':
            return 1
        else:
            boundaries = self._call(t).boundaries()
            return np.linalg.norm(np.max(np.abs(boundaries - 1), axis=1), ord=2)


class ConstantDynamics(PIDynamics):
    """ Time-independent, price-independent price dynamics.
    It is a price dynamics where increments (for additive) or multipliers (for multiplicative)
    don't depend both on previous prices and time.

    Parameters
    ----------
    support: ISetHandler
        Set of multipliers (or increments, if model is additive)
    type: {'mult', 'add'}
        Dynamics type: multiplicative or additive

    Notes
    -----
    This class, essentially, functions as IdenticalMap with additional properties `type` and `t_max`.
    """

    def __init__(self, support: ISetHandler, type='mult'):
        self._support = support
        if type not in self._allowed_types:
            raise TypeError('Wrong type dynamics!')
        self._type = type

    def _call(self, t):
        return self._support

    @property
    def dim(self):
        return self._support.dim

    @property
    def type(self):
        return self._type

    @property
    def t_max(self):
        """:code:`np.inf` (since the model is time-independent)"""
        return np.inf


class MDAFDynamics(PriceDynamics):
    r"""Multiplicative dynamics in additive form (time-independent).

    Given as its support a set of `price-independent` multipliers :math:`K`, it returns (when called), a set of `increments` :math:`\{y = k\cdot x - x:\; k \in K\}`.
    Here, :math:`k \cdot x` denotes `element-wise` multiplication (Hadamard product) of :math:`k` and :math:`x`.
    """

    def __init__(self, support: ISetHandler):
        self._support = support

    def __call__(self, x, t=1):
        return self._support.multiply(x).add(-x)

    @property
    def dim(self):
        return self._support.dim

    @property
    def type(self):
        return 'add'

    @property
    def t_max(self):
        return np.inf

    def get_lipschitz(self, t: int):
        return np.linalg.norm(np.max(np.abs(self._support.boundaries() - 1), axis=1), ord=2)


# Two simple constraints for ease of use
NoConstraints = IdenticalMap(RealSpaceHandler())
LongOnlyConstraints = IdenticalMap(NonNegativeSpaceHandler())
