# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
r""" This submodule implements a :class:`Lattice` class for working with n-dimensional lattices on :math:`\mathbb{R}^{n}`

The rationale behind such lattice is that it is much easier to work with integer coordinates than with real ones.
:class:`Lattice` incapsulates such a uniform lattice.

"""

import numpy as np

from ..util import coalesce, cartesian_product
from .set_handler import ISetHandler

__all__ = ['Lattice']


class Lattice:
    r""" Handler for the uniform or logscale n-dimensional lattice. Provides functionality
    for mapping integer point coordinates to :math:`\mathbb{R}^{n}` and vice versa.
    
    Parameters
    ----------
    delta : array_like, size = n x 1
        Lattice steps.
    logscale : bool, default = False
        If True, the lattice is considered logscale. Default is False.
    center : array_like, size = n x 1
        Point from :math:`\mathbb{R}^{n}` which corresponds to zero coordinates on a lattice.
    dtype : numeric np.dtype
        Type for points in :math:`\mathbb{R}^{n}`. Default is np.float64.
    dtype_p : numeric np.dtype
        Type for lattice coordinates. Default is np.int64.



    Notes
    ------
    The class upholds the following notation: 'x' means points from :math:`\mathbb{R}^{n}`,
    'point' means points from the lattice.
    
    """

    def __init__(self, delta, logscale=False, center=None, dtype=None, dtype_p=None):

        self.dtype = coalesce(dtype, np.float64)
        self.dtype_p = coalesce(dtype_p, np.int64)
        self.delta = np.asarray(delta, dtype=self.dtype)
        self.logscale = logscale
        self.center = np.asarray(coalesce(center, self.delta * 0), dtype=self.dtype)

    @property
    def dim(self):
        """ Returns lattice's dimension

        Returns
        -------
        int
            Lattice's dimesion

        """
        return self.delta.shape[0]

    def _x_trans(self, x):
        r""" Utility function for reducing logscale lattice logic to uniform lattice.

        Parameters
        ----------
        x : array_like
            Points from :math:`\mathbb{R}^{n}`.

        Returns
        -------
        np.ndarray
            logarithm of `x` if the lattice is logscale, otherwise `x`.

        """

        return x if self.logscale == False else np.log(x)

    def _x_trans_inv(self, x):
        """ Inverse transform to :meth:`_x_trans()`.

        """

        return x if self.logscale == False else np.exp(x)

    def get_projection(self, obj):
        r""" Projects a point or :math:`\mathbb{R}^{n}`, an array of points or a predefined set to lattice coordinates.

        Parameters
        ----------
        obj : array_like, size = (n,) or (m,n), or ISetHandler
            A set of points from :math:`\mathbb{R}^{n}`.

        Returns
        -------
        np.ndarray
            If obj is an array, returns an array of coordinates.
            If obj is an ISetHandler instance, the ISetHandler method of lattice-projection will be used
            to return the projection of the set to the lattice.

        """

        if isinstance(obj, ISetHandler):  # obj is a constraint or support set

            return obj.project(self)

        else:  # obj is an array of coordinates

            return self._get_point(obj)

    def _get_point(self, x):
        r""" Returns lattice coordinates of the array of points from :math:`\mathbb{R}^{n}`.

        Parameters
        ----------
        x : array_like, size = (n,) or (m,n)
            A set of points from :math:`\mathbb{R}^{n}`.

        Returns
        -------
        np.ndarray
            An array of lattice coordinates.

        """

        x = self._x_trans(np.asarray(x, dtype=self.dtype))

        return np.rint((x - self.center) / self.delta).astype(self.dtype_p)

    def map2x(self, point):
        r""" Maps the array of lattice coordinates to points in :math:`\mathbb{R}^{n}`.

        This function acts as a sort of inverse to :meth:`get_projection()`.

        Parameters
        ----------
        point : array_like, size = (n,) or (m,n)
            A set of integer-based coordinates on the lattice.

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
        """ Returns lattice projection for the parallelotope specified
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
            An array of lattice coordinates.

        See Also
        --------
        :func:`guaranteed.util.util.cartesian_product` : Function used for generation of points from corner coordinates

        """

        p_from = self._get_point(x_from)
        p_to = self._get_point(x_to)

        return cartesian_product(*[np.arange(p[0], p[1] + 1) for p in zip(np.atleast_1d(p_from), np.atleast_1d(p_to))])
