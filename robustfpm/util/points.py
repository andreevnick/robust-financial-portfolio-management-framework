# Copyright 2021 portfolio-robustfpm-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import sys
import numpy as np

from .util import pairing_function, cartesian_product


__all__ = [
            'unique_points_union',
            'minksum_points',
            'minkprod_points',
            'isin_points',
            'setdiff_points',
            'square_neighbourhood_on_lattice',
            'diamond_neighbourhood_on_lattice'
          ]

""" The module provides operations over the sets of points on the n-dimensional lattice.
Points are represented as n-dimensional integer vectors of coordinates on the lattice.
"""
    

def unique_points_union(points1, points2):
    """
    Return the union of two point sets.
    
    Parameters
    ----------
    points1 : array-like
        A set of unique lattice points.
    points2 : array-like
        A set of unique lattice points.
        
    Returns
    -------
    numpy.ndarray
        Union of points1 and points2.
    
    """
    
    points1 = np.atleast_2d(points1)
    points2 = np.atleast_2d(points2)
    
    if points1.shape[0] == 0:
        return points2
    
    if points2.shape[0] == 0:
        return points1
    
    return np.vstack((points1, points2[np.in1d(pairing_function(points2), pairing_function(points1),
                                              assume_unique=True, invert=True)]))
    
    
def __minksum_points(points_iter, points_sum):

    if points_iter.shape[0] == 1:
        return points_iter[0] + points_sum
        
        
    res = np.ndarray((0, points_sum.shape[1]), dtype = points_sum.dtype)
        
    for point in points_iter:
    
        res = unique_points_union(res, point + points_sum)
        
    return res


def minksum_points(points1, points2, recur_max_level=None):
    """
    Return the Minkowski sum of two point sets.
    
    Parameters
    ----------
    points1 : array-like
        A set of lattice points.
    points2 : array-like
        A set of lattice points.
    recur_max_level : int
        Max recursion level for the numeric algorithm. When None, half of
        the maximum allowed recursion level is used.
        
    Returns
    -------
    numpy.ndarray
        Minkowski sum of points1 and points2.
    
    """
    
    if recur_max_level is None:
        recur_max_level = sys.getrecursionlimit() // 2
    
    if recur_max_level <= 1:
        return __minksum_points(points1, points2)
    
    else:
        if points1.shape[0] < points2.shape[0]:
            points_iter = points1
            points_sum = points2

        else:
            points_iter = points2
            points_sum = points1

        n = points_iter.shape[0]
        
        if n == 0:
            return points_sum
        
        elif n == 1:
            return __minksum_points(points_iter, points_sum)
        
        else:
            return unique_points_union(minksum_points(points_iter[:(n//2),:], points_sum, recur_max_level=recur_max_level-1),
                                       minksum_points(points_iter[(n//2):,:], points_sum, recur_max_level=recur_max_level-1)
                                      )


def isin_points(points1, points2):
    """
    :code:`numpy.in1d` for sets of unique points
    
    Parameters
    ----------
    points1 : array-like
        A set of unique lattice points.
    points2 : array-like
        A set of unique lattice points.
        
    Returns
    -------
    numpy.ndarray
        See numpy.in1d
    
    """
    
    return np.in1d(pairing_function(points1), pairing_function(points2), assume_unique=True)


def setdiff_points(pointsA, pointsB):
    """
    Return set difference for sets of unique points
    
    Parameters
    ----------
    pointsA : array-like
        A set of unique lattice points.
    pointsB : array-like
        A set of unique lattice points.
        
    Returns
    -------
    array-like
        Points  from pointsA which are not in pointsB.
    
    """
    
    return pointsA[np.in1d(pairing_function(pointsA), pairing_function(pointsB), assume_unique=True, invert=True)]


def square_neighbourhood_on_lattice(lattice_point, radius, include_center=False):
    """
    Return a lattice set, representing the Moore (square) neighbourhood of the specified point.

    Square neighbourhood (denoted by ``X``) of a point (denoted by ``*``): ::

        O|O|O|O|O
        ---------
        O|X|X|X|O
        ---------
        O|X|*|X|O
        ---------
        O|X|X|X|O
        ---------
        O|O|O|O|O
    
    Parameters
    ----------
    lattice_point : array-like
        A lattice point.
    radius : int >= 0
        Radius of the neighbourhood.
    include_center : bool
        If True, lattice_point is included in the returned set. Default is False.
        
    Returns
    -------
    array-like
        Set of lattice points, representing a square neighbourhood with the specified radius.
    
    """
    
    lattice_point = np.asarray(lattice_point)
    
    U = np.asarray(cartesian_product(*[np.arange(p-radius, p+radius+1) for p in lattice_point]))
    
    if include_center:
        return U
    else:
        return np.asarray([p for p in U if np.any(p-lattice_point)])


def diamond_neighbourhood_on_lattice(lattice_point, radius, include_center=False):
    """
    Return a lattice set, representing the von Neumann (diamond) neighbourhood of the specified point.

    Diamond neighbourhood (denoted by ``X``) of a point (denoted by ``*``): ::

        O|O|O|O|O
        ---------
        O|O|X|O|O
        ---------
        O|X|*|X|O
        ---------
        O|O|X|O|O
        ---------
        O|O|O|O|O
    
    Parameters
    ----------
    lattice_point : array-like
        A lattice point.
    radius : int >= 0
        Radius of the neighbourhood.
    include_center : bool
        If True, lattice_point is included in the returned set. Default is False.
        
    Returns
    -------
    array-like
        Set of lattice points, representing a von Neumann neighbourhood with the specified radius.
    
    """
    
    lattice_point = np.asarray(lattice_point)
    
    return np.asarray([p for p in square_neighbourhood_on_lattice(lattice_point, radius, include_center) if np.sum(np.abs(lattice_point-p)) <= radius])


def __minkprod_points(lattice, points, set_handler, pos):
    """
    set_handler should represent a non-negative set
    """
    
    points_ext = minksum_points(points, square_neighbourhood_on_lattice(np.zeros_like(lattice.delta), 1, include_center=True))
    
    if pos:
        x = lattice.map2x(points_ext)
        points_ext = points_ext[np.min(x, axis=1) > 0, :]
    
    set_sum = np.empty((0, points_ext.shape[1]), dtype = points_ext.dtype)
    
    for p in points_ext:
        
        s = set_handler.multiply(lattice.map2x(p)).project(lattice)
        
        set_sum = unique_points_union(set_sum, s)
        
    if pos:
        x = lattice.map2x(set_sum)
        set_sum = set_sum[np.min(x, axis=1) > 0, :]
        
    return set_sum


def minkprod_points(lattice, points, set_handler, pos=False, recur_max_level=None):
    r"""
    Calculate the set of element-wise products of two sets.
    
    For the provided set handler and the set of points on the lattice calculates
    the approximation of the set of `element-wise` (Hadamard) products
    :math:`\{x\cdot y\}`, where :math:`x` is from the set, :math:`y` is from the set handler projection on the lattice.
    Optionally filters out points of the product which have non-positive coordinates.
    
    Parameters
    ----------
    
    lattice : class:`lattice`
        Point lattice.
    points : array-like
        Sequence of points on the lattice.
    set_handler : class:`ISetHandler`
        Set handler, representing a non-negative set.
    pos : bool
        If True, the points of the product with the non-positive coordinates
        will be filtered out from the result.
    recur_max_level : int
        Max recursion level for the numeric algorithm. When None, half of
        the maximum allowed recursion level is used.
    
    Returns
    -------
    numpy.ndarray
        Set of element-wise products of the provided sets.
        
    """
    
    points = np.atleast_2d(points)
    
    if recur_max_level is None:
        recur_max_level = sys.getrecursionlimit() // 2
    
    if recur_max_level <= 1:
        return __minkprod_points(lattice, points, set_handler, pos)
    
    else:

        n = points.shape[0]
        
        if n == 0:
            return []
        
        if n == 1:
            return __minkprod_points(lattice, points, set_handler, pos)
        
        else:
            return unique_points_union(minkprod_points(lattice, points[:(n//2),:], set_handler, pos=pos, recur_max_level=recur_max_level-1),
                                       minkprod_points(lattice, points[(n//2):,:], set_handler, pos=pos, recur_max_level=recur_max_level-1)
                                      )


