# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import numpy as np

from ..util import coalesce
from .set_handler import ISetHandler


__all__ = ['Grid']


class Grid:
    """ Handler for the uniform or logscale n-dimensional grid. Privides functionality
    for mapping integer point coordinates to R^n and vice versa.
    
    Parameters
    ----------
    delta : array-like, size = n x 1
        Grid steps.
    logscale : boolean
        If True, the grid is considered logscale. Default is False.
    center : array-like, size = n x 1
        Point from R^n which corresponds to zero coordinates on a grid.
    dtype : numpy type
        Type for points in R^n. Default is np.float64.
    dtype_p : numpy type
        Type for grid coordinates. Default is np.int64.
        
    Note:
    ------
    The class upholds the following notation: 'x' means points from R^n,
    'point' means points from the grid.
    
    """
    
    def __init__(self, delta, logscale=False, center=None, dtype=None, dtype_p=None):
        
        self.dtype = coalesce(dtype, np.float64)
        self.dtype_p = coalesce(dtype_p, np.int64)
        self.delta = np.asarray(delta, dtype=self.dtype)        
        self.logscale = logscale        
        self.center = np.asarray(coalesce(center, self.delta*0), dtype=self.dtype)
        
    def x_trans(self, x):
        """ Utility function for reducing logscale grid logic to uniform grid.

        Parameters
        ----------
        x : array-like
            Points from R^n.

        Returns
        -------
        numpy.ndarray
            logarithm of x if the grid is logscale, otherwise x.

        """
    
        return x if self.logscale == False else np.log(x)
    
    
    def x_trans_inv(self, x):
        """ Reverse transform to x_trans().

        """
        
        return x if self.logscale == False else np.exp(x)

        
    def get_projection(self, obj, **kwargs):
        """ Projects a point or R^n, an array of points or a predefined set to grid coordinates.

        Parameters
        ----------
        obj : array-like, size = (n,) or (m,n), or an ISetHandler object
            A set of points from R^n.
        kwargs : deprecated

        Returns
        -------
        numpy.ndarray
            If obj is an array, returns an array of coordinates. If obj implements an ISetHandler
            interface, the ISetHandler method of grid-projection will be used to return the projection
            of the set to the grid.

        """
        
        if isinstance(obj, ISetHandler): # obj is a constraint or support set
            
            return obj.project(self)
        
        else: # obj is an array of coordinates
            
            return self.get_point(obj)
        
        
    def get_point(self, x):
        """ Returns grid coordinates of the array of points from R^n.

        Parameters
        ----------
        obj : array-like, size = (n,) or (m,n)
            A set of points from R^n.

        Returns
        -------
        numpy.ndarray
            An array of coordinates.

        """
        
        x = self.x_trans(np.asarray(x, dtype=self.dtype))
        
        return np.rint((x - self.center)/self.delta).astype(self.dtype_p)
    
    
    def map2x(self, point):
        """ Maps the array of grid coordinates to points in R^n.

        Parameters
        ----------
        point : array-like, size = (n,) or (m,n)
            A set of integer-based coordinates on the grid.

        Returns
        -------
        numpy.ndarray
            An array of points in R^n.

        """
                    
        return self.x_trans_inv(point * self.delta + self.center)
    
    
    def xrectangle_points(self, x_from, x_to):
        """ Returns projection to the grid for the hyperrectangle specified
        via its corner points.

        Parameters
        ----------
        x_from : array-like, size = (n,)
            'Lower left' corner of the hyperrectangle.
        x_to : array-like, size = (n,)
            'Upper right' corner of the hyperrectangle.

        Returns
        -------
        numpy.ndarray
            An array of coordinates.

        """
        
        p_from = self.get_point(x_from)
        p_to = self.get_point(x_to)
        
        return cartesian_product(*[np.arange(p[0], p[1]+1) for p in zip(np.atleast_1d(p_from), np.atleast_1d(p_to))])
        
        