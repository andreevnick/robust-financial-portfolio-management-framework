# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
r""" This submodule implements a Grid class for working with n-dimensional grids on :math:`\mathbb{R}^{n}`

The rationale behind such grid is that it is much easier to work with integer coordinates than with real ones.
:class:`Grid` incapsulates such a uniform grid.

"""

import numpy as np

from ..util import coalesce, cartesian_product
from .set_handler import ISetHandler

__all__ = ['Grid']


class Grid:
    r""" Handler for the uniform or logscale n-dimensional grid. Provides functionality
    for mapping integer point coordinates to :math:`\mathbb{R}^{n}` and vice versa.
    
    Parameters
    ----------
    delta : array_like, size = n x 1
        Grid steps.
    logscale : bool, default = False
        If True, the grid is considered logscale. Default is False.
    center : array_like, size = n x 1
        Point from :math:`\mathbb{R}^{n}` which corresponds to zero coordinates on a grid.
    dtype : numeric np.dtype
        Type for points in :math:`\mathbb{R}^{n}`. Default is np.float64.
    dtype_p : numeric np.dtype
        Type for grid coordinates. Default is np.int64.



    Notes
    ------
    The class upholds the following notation: 'x' means points from :math:`\mathbb{R}^{n}`,
    'point' means points from the grid.
    
    """

    def __init__(self, delta, logscale=False, center=None, dtype=None, dtype_p=None):

        self.dtype = coalesce(dtype, np.float64)
        self.dtype_p = coalesce(dtype_p, np.int64)
        self.delta = np.asarray(delta, dtype=self.dtype)
        self.logscale = logscale
        self.center = np.asarray(coalesce(center, self.delta * 0), dtype=self.dtype)

    def _x_trans(self, x):
        """ Utility function for reducing logscale grid logic to uniform grid.

        Parameters
        ----------
        x : array_like
            Points from :math:`\mathbb{R}^{n}`.

        Returns
        -------
        np.ndarray
            logarithm of `x` if the grid is logscale, otherwise `x`.

        """

        return x if self.logscale == False else np.log(x)

    def _x_trans_inv(self, x):
        """ Inverse transform to :meth:`_x_trans()`.

        """

        return x if self.logscale == False else np.exp(x)

    def get_projection(self, obj):
        r""" Projects a point or :math:`\mathbb{R}^{n}`, an array of points or a predefined set to grid coordinates.

        Parameters
        ----------
        obj : array_like, size = (n,) or (m,n), or ISetHandler
            A set of points from :math:`\mathbb{R}^{n}`.

        Returns
        -------
        np.ndarray
            If obj is an array, returns an array of coordinates.
            If obj is an ISetHandler instance, the ISetHandler method of grid-projection will be used
            to return the projection of the set to the grid.

        """

        if isinstance(obj, ISetHandler):  # obj is a constraint or support set

            return obj.project(self)

        else:  # obj is an array of coordinates

            return self._get_point(obj)

    def _get_point(self, x):
        """ Returns grid coordinates of the array of points from :math:`\mathbb{R}^{n}`.

        Parameters
        ----------
        obj : array_like, size = (n,) or (m,n)
            A set of points from :math:`\mathbb{R}^{n}`.

        Returns
        -------
        np.ndarray
            An array of grid coordinates.

        """

        x = self._x_trans(np.asarray(x, dtype=self.dtype))

        return np.rint((x - self.center) / self.delta).astype(self.dtype_p)

    def map2x(self, point):
        """ Maps the array of grid coordinates to points in :math:`\mathbb{R}^{n}`.

        This function acts as a sort of inverse to :meth:`get_projection()`.

        Parameters
        ----------
        point : array_like, size = (n,) or (m,n)
            A set of integer-based coordinates on the grid.

        Returns
        -------
        np.ndarray
            An array of points in :math:`\mathbb{R}^{n}`.

        Notes
        -----
        get_projection(map2x(`point`) = `point`, but map2x(get_projection(`x`)) in general is not equal to `x` (due to mapping accuracy).
        """

        return self._x_trans_inv(point * self.delta + self.center)

    def xrectangle_points(self, x_from, x_to):
        """ Returns projection to the grid for the parallelotope specified
        via its corner points.

        Parameters
        ----------
        x_from : array_like, size = (n,)
            'Lower left' corner of the parallelotope.
        x_to : array_like, size = (n,)
            'Upper right' corner of the parallelotope.

        Returns
        -------
        np.ndarray
            An array of grid coordinates.

        See Also
        --------
        :func:`guaranteed.util.util.cartesian_product` : Function used for generation of points from corner coordinates

        """

        p_from = self._get_point(x_from)
        p_to = self._get_point(x_to)

        return cartesian_product(*[np.arange(p[0], p[1] + 1) for p in zip(np.atleast_1d(p_from), np.atleast_1d(p_to))])
