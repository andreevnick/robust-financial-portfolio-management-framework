# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

""" This submodule implements :class:`ISetHandler` as an abstract interface for set and different types of sets with
respective methods.

"""

import numpy as np
from ..util import coalesce, cartesian_product
from abc import ABC, abstractmethod

__all__ = [
    'ISetHandler',
    'RectangularHandler',
    'EllipseHandler',
    'RealSpaceHandler',
    'NonNegativeSpaceHandler'
]


class ISetHandler(ABC):
    """ Interface which contains the set-based operations for implementation in non-abstract SetHandlers.
    
    """

    @abstractmethod
    def project(self, grid):
        """ Projects the set onto the grid.

        Parameters
        ----------
        grid : Grid
            Target grid.

        Returns
        -------
        np.ndarray
            Array of points on the grid which belong to the set.

        """

        raise NotImplementedError('The method must be defined in a subclass')

    @abstractmethod
    def support_function(self, x):
        """ Returns value of the support function of the set at point `x`.

        Parameters
        ----------
        x : np.ndarray, size = (m,n)
            Array of points from :math:`\mathbb{R}^{n}`.

        Returns
        -------
         np.ndarray
            Array of support function values, may be np.Inf.

        Notes
        -----
        The support function :math:`\sigma_A(l)` of any non-empty subset :math:`A \subseteq X`,
         where :math:`X` is a Banach space, is defined by the following relation:

         .. math:: \sigma_A(l) = \sup\limits_{a \in A} \langle a, l \\rangle,

         for each :math:`l \in X^*` — continuous dual space of X


        """

        raise NotImplementedError('The method must be defined in a subclass')

    @abstractmethod
    def iscompact(self):
        """ Checks if the set is a `compact <https://en.wikipedia.org/wiki/Compact_space>`_
        subset of :math:`\mathbb{R}^{n}`.

        Returns
        -------
        boolean
            True if the set is compact.

        """

        raise NotImplementedError('The method must be defined in a subclass')

    @abstractmethod
    def multiply(self, x):
        """ Multiplies the set by a point `x` from :math:`\mathbb{R}^{n}`.

        Parameters
        ----------
        x : np.ndarray
            Point to multiply the set by

        Returns
        -------
        ISetHandler
            Set handler that corresponds to the set :math:`\{x\cdot a, a \in A\}`, where :math:`A` is the current set (`self`).

        """

        raise NotImplementedError('The method must be defined in a subclass')

    @abstractmethod
    def add(self, x):
        """ Adds a point `x` from :math:`\mathbb{R}^{n}` to the set.

        Parameters
        ----------
        x : np.ndarray
            Point to add to the set

        Returns
        -------
        ISetHandler
            Set handler that corresponds to the set :math:`\{x + a, a \in A\}`, where :math:`A` is the current set (`self`).

        """

        raise NotImplementedError('The method must be defined in a subclass')


class RectangularHandler(ISetHandler):
    """ Handler for a rectangular set.

    Such a set :math:`\mathcal{X}\subseteq \mathbb{R}^{n}` is defined as:

    .. math :: \mathcal{X} = \{(x_1,\dots,x_n) \in \mathbb{R}^{n}:\; lb_i \leqslant x_i \leqslant ub_i, i = 1,2,\dots,n\},

    where :math:`lb_i, ub_i` are given bounds on i-th dimension.

    Parameters
    ----------
    bounds : array_like, size = (2,) or (n,2)
        Array of boundary points [:math:`lb_i, ub_i`]
    dtype : np.dtype, optional
        Numpy datatype of points in bounds

    """

    def __init__(self, bounds, dtype=None):
        self.bounds = np.atleast_2d(np.asarray(bounds, dtype=coalesce(dtype, np.float64))).reshape(-1,2)

    def project(self, grid):
        bnd = np.vstack((grid.get_projection(self.bounds.T[0]), grid.get_projection(self.bounds.T[1]))).T

        return cartesian_product(*[np.arange(lb, ub + 1) for lb, ub in bnd])

    def support_function(self, x):
        x = np.atleast_2d(x)

        return np.sum(np.maximum(x * np.tile(self.bounds.T[0], (x.shape[0], 1)),
                                 x * np.tile(self.bounds.T[1], (x.shape[0], 1))),
                      axis=1)

    def iscompact(self):
        return not np.any(np.isinf(self.bounds))

    def multiply(self, x):
        assert np.all(x >= 0), 'x must be >= 0'
        mul = np.asarray(x).reshape(-1, 1)
        return RectangularHandler(self.bounds * mul, dtype=self.bounds.dtype)

    def add(self, x):
        to_add = np.asarray(x).reshape(-1, 1)
        return RectangularHandler(self.bounds + to_add, dtype=self.bounds.dtype)


class EllipseHandler(ISetHandler):
    """ Handler for an n-dimensional ellipsoid

    """

    def __init__(self, mu, sigma, conf_level=None, dtype=None):
        self.mu = np.atleast_2d(mu).astype(coalesce(dtype, np.float64))

        self.sigma = np.atleast_2d(sigma).astype(coalesce(dtype, np.float64))

        if conf_level is not None:
            # https://stats.stackexchange.com/questions/64680/how-to-determine-quantiles-isolines-of-a-multivariate-normal-distribution
            self.sigma *= -2 * np.log(1 - conf_level)

        eigv, U = np.linalg.eig(self.sigma)
        self.L = np.sqrt(np.diag(eigv))
        self.U = U

        self.sigma_inv = np.linalg.inv(self.sigma)

    def __r2(self, x):
        return np.sum((x - self.mu).dot(self.sigma_inv) * (x - self.mu), axis=1)

    def project(self, grid):
        R = np.max(np.abs(np.diagonal(self.L))) + np.max(grid.delta)

        S = RectangularHandler(self.mu.T + np.array([-R, R], dtype=self.L.dtype)).project(grid)

        return S[self.__r2(grid.map2x(S)) <= 1]

    def support_function(self, x, x_center=None):
        x = np.atleast_2d(x)

        return (x.T @ self.mu) + np.linalg.norm(self.L @ self.U.T @ x.T, axis=0, ord=2)

    def iscompact(self):
        return (not np.any(np.isinf(self.sigma))) and (not np.any(np.isinf(self.mu)))

    def multiply(self, x):

        assert np.all(x >= 0), 'x must be >= 0'

        D = np.diag(x)

        return EllipseHandler(self.mu @ D, D @ self.sigma @ D.T, dtype=np.result_type(self.mu, self.sigma))

    def add(self, x):
        return EllipseHandler(self.mu + x.reshape(1, -1), self.sigma, dtype=np.result_type(self.mu, self.sigma))


class RealSpaceHandler(ISetHandler):
    """ Handler for :math:`\mathbb{R}^{n}`

    """

    def __init__(self):
        pass

    def project(self, grid):
        raise Exception('Cannot map the unbounded set')

    def support_function(self, x):
        x = np.atleast_2d(x)

        return np.where(np.all(x == 0, axis=1), 0, np.Inf)

    def iscompact(self):
        return False

    def multiply(self, x):
        return RealSpaceHandler()

    def add(self, x):
        return RealSpaceHandler()


class NonNegativeSpaceHandler(ISetHandler):
    r""" Handler for closed non-negative orthant of :math:`\mathbb{R}^{n}`
    Such an orthant is defined as

    .. math:: \mathbb{R}^{n}_{+} = \{(x_1,\dots,x_n) \in \mathbb{R}^n:\; x_i \geqslant 0, i = 1,2,\dots,n\}

    """

    def __init__(self):
        pass

    def project(self, grid):
        raise Exception('Cannot map the unbounded set')

    def support_function(self, x):
        x = np.atleast_2d(x)

        return np.where(np.all(x <= 0, axis=1), 0, np.Inf)

    def iscompact(self):
        return False

    def multiply(self, x):
        return NonNegativeSpaceHandler()

    def add(self, x):
        raise NotImplementedError('Addition to NonNegativeSpaceHandler is not implemented.')
