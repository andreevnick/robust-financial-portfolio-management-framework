import sys
import numpy as np

from .util import pairing_function, cartesian_product


__all__ = [
            'unique_points_union',
            'minksum_points',
            'minkprod_points',
            'isin_points',
            'setdiff_points',
            'square_neighbourhood_on_grid',
            'diamond_neighbourhood_on_grid'
          ]

    

def unique_points_union(points1, points2):
    
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
    
    return np.in1d(pairing_function(points1), pairing_function(points2), assume_unique=True)


def setdiff_points(pointsA, pointsB):
    
    return pointsA[np.in1d(pairing_function(pointsA), pairing_function(pointsB), assume_unique=True, invert=True)]


def square_neighbourhood_on_grid(grid_point, radius, include_center=False):
    
    grid_point = np.asarray(grid_point)
    
    U = np.asarray(cartesian_product(*[np.arange(p-radius, p+radius+1) for p in grid_point]))
    
    if include_center:
        return U
    else:
        return np.asarray([p for p in U if np.any(p-grid_point)])


def diamond_neighbourhood_on_grid(grid_point, radius, include_center=False):
    
    grid_point = np.asarray(grid_point)
    
    return np.asarray([p for p in square_neighbourhood_on_grid(grid_point, radius, include_center) if np.sum(np.abs(grid_point-p)) <= radius])


def __minkprod_points(grid, points, set_handler, pos):
    '''
    set_handler should represent a non-negative set
    '''
    
    points_ext = minksum_points(points, square_neighbourhood_on_grid([0,0], 1, include_center=True))
    
    if pos:
        x = grid.map2x(points_ext)
        points_ext = points_ext[np.min(x, axis=1) > 0, :]
    
    set_sum = np.empty((0, points_ext.shape[1]), dtype = points_ext.dtype)
    
    for p in points_ext:
        
        s = set_handler.multiply(grid.map2x(p)).project(grid)
        
        set_sum = unique_points_union(set_sum, s)
        
    if pos:
        x = grid.map2x(set_sum)
        set_sum = set_sum[np.min(x, axis=1) > 0, :]
        
    return set_sum


def minkprod_points(grid, points, set_handler, pos=False, recur_max_level=None):
    
    points = np.atleast_2d(points)
    
    if recur_max_level is None:
        recur_max_level = sys.getrecursionlimit() // 2
    
    if recur_max_level <= 1:
        return __minkprod_points(grid, points, set_handler, pos)
    
    else:

        n = points.shape[0]
        
        if n == 0:
            return []
        
        if n == 1:
            return __minkprod_points(grid, points, set_handler, pos)
        
        else:
            return unique_points_union(minkprod_points(grid, points[:(n//2),:], set_handler, pos=pos, recur_max_level=recur_max_level-1),
                                       minkprod_points(grid, points[(n//2):,:], set_handler, pos=pos, recur_max_level=recur_max_level-1)
                                      )


