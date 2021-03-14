# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

r""" This submodule implements :class:`MultivalMap` as an abstract interface for multivalued mappings and some concrete mappings.

For any sets :math:`X, Y`, a multivalued mapping is such a mapping :math:`\Gamma: X \mapsto Y`, that

.. math :: \forall x \in X \; \Gamma(x) \subseteq Y

"""

import numpy as np
from abc import ABC, abstractmethod
from .set_handler import ISetHandler
from .grid import Grid

__all__ = ['IMultivalMap', 'IdenticalMap', 'PriceDynamics', 'TIMIDynamics']


class IMultivalMap(ABC):
    """ An abstract interface class for multivalued mappings"""

    @abstractmethod
    def __call__(self, x, t):
        """ A call of a multivalued mapping

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


class IdenticalMap(IMultivalMap):
    """ Identical mapping: always returns the same set (its support)

    Parameters
    ----------
    support: ISetHandler
        The underlying set to return
    """

    def __init__(self, support: ISetHandler):
        self.support = support

    def __call__(self, x, t):
        return self.support


class PriceDynamics(IMultivalMap):
    """An abstract base class for different price dynamics
    """
    _allowed_types = {'mult', 'add'}

    @abstractmethod
    def __call__(self, x, t=1):
        """

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
        """

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


class TIMIDynamics(PriceDynamics):
    """ Time-independent, multiplicatively-independent price dynamics. That is,

    .. math :: K_t(x_0, \dots, x_{t-1}) = \{(y_1, \dots, y_n) \in \mathbb{R}^n:\; y_i = (m_i - 1)\cdot x_i,\; (m_1,\dots, m_n) \in C^*\},

    where :math:`C^* \subseteq \mathbb{R}^n` is constant.
    In other words, it is essentially a multiplicative dynamics where multipliers don't depend both on previous prices and time.

    Parameters
    ----------
    support: ISetHandler
        Set of multipliers (or increments, if model is additive)
    type: {'mult', 'add'}
        Dynamics type: multiplicative or additive

    Notes
    -----
    This class, essentially, functions as IdenticalMap with additional property `type`.
    """

    def __init__(self, support: ISetHandler, type='mult'):
        self._support = support
        if type not in self._allowed_types:
            raise TypeError('Wrong type dynamics!')
        self._type = type

    def __call__(self, x, t=1):
        return self._support

    @property
    def type(self):
        return self._type

    @property
    def t_max(self):
        """np.inf (since the model is time-independent)"""
        return np.inf

    def get_lipschitz(self, t: int):
        # TODO: implement this method
        pass
