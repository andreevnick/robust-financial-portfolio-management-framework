# Copyright 2021 portfolio-robustfpm-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import numpy as np

from sklearn.utils import check_random_state

from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning
from scipy.linalg import LinAlgWarning
from scipy.spatial import ConvexHull

import warnings

__all__ = ['generate_vertices_rectangle',
           'distance',
           'nearest',
           'farthest',
           'pop_random',
           'get_max_coordinates',
           'in_hull'
           ]


def generate_vertices_rectangle(size, dim, center=0, random_state=None):
    if random_state is None:
        random_state = check_random_state(random_state)

    return list(random_state.uniform(size=(size, dim)).astype(np.float64) - 0.5)


def distance(p1, p2, p=2):
    return np.linalg.norm(p1 - p2, ord=p)


def nearest(p, S):
    d = np.Inf

    for i, curr_point in enumerate(S):

        curr_d = distance(p, curr_point)

        if curr_d < d:
            d = curr_d
            ind = i

    return (ind, d)


def farthest(p, S):
    d = -np.Inf

    for i, curr_point in enumerate(S):

        curr_d = distance(p, curr_point)

        if curr_d > d:
            d = curr_d
            ind = i

    return (ind, d)


def pop_random(element_list, random_state=None):
    if random_state is None:
        random_state = check_random_state(random_state)

    n = len(element_list)

    if n == 0:
        raise ValueError('the list is empty')

    return element_list.pop(np.random.randint(n))


def get_max_coordinates(x, f, z, method='interior-point', tol=1e-8, debug_mode=False, ignore_warnings=False):
    """ Solves the problem

    .. math:: p_1 \cdot f_1 + ... + p_m \cdot f_m \\rightarrow \max,
    .. math:: p_1 \cdot x_1 + ... p_m \cdot x_m = z,
    .. math:: p_1 + ... + p_m = 1,
    .. math:: p_1, ... p_m \geqslant 0

    w.r.t. m-dimensional vector p and returns the optimal p. Equivalently, finds the barycentric coordinates
    of the points of the facet, which belongs to the concave hull of the graph of f and contains z when
    projected.
    
    Parameters
    ----------
    x, f : numpy.ndarray
        Points and function values of the graph.
    z : numpy.ndarray.
        Point with the same dimansion as points in x. Must be in the convex hull of x.
        
    Returns
    -------
    numpy.ndarray
        Barycentric coordinates p.
    
    """

    m = f.size
    c = -f
    A = np.vstack((np.ones(m), x.T))
    b = np.insert(z, 0, float(1))

    opts = {'tol': tol}

    if debug_mode:
        opts['disp'] = True

    if ignore_warnings:

        with warnings.catch_warnings():

            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", LinAlgWarning)

            res = linprog(c, A_eq=A, b_eq=b, bounds=(float(0), None), method=method, options=opts)

    else:

        res = linprog(c, A_eq=A, b_eq=b, bounds=(float(0), None), method=method, options=opts)

    if res.status > 0:

        if debug_mode:
            print(res)
            print('z = ', z)
            print('c = ', c)
            print('A = ', A)
            print('b = ', b)
            print('method = ', method)

        raise RuntimeError('Unable to find the convex representation of z!')

    else:
        return res.x


def is_coplanar_2D(x_array, tol=1e-12):
    return np.abs(np.linalg.det(np.vstack((x_array[1] - x_array[0], x_array[2] - x_array[0])))) <= tol


def is_between_2D(z, x0, x1, tol=1e-12):
    alpha = ((z - x1) @ (x0 - x1)) / ((x0 - x1) @ (x0 - x1))

    return (alpha >= 0) and (alpha <= 1) and (np.max(np.abs(z - x1 - alpha * (x0 - x1))) <= tol)


def in_triangle_2D(z, points, tol=1e-12):
    if is_coplanar_2D(points, tol=tol):

        for ind in [[1, 2], [0, 2], [0, 1]]:

            alpha = ((z - points[ind[1]]) @ (points[ind[0]] - points[ind[1]])) / (
                        (points[ind[0]] - points[ind[1]]) @ (points[ind[0]] - points[ind[1]]))

            if (alpha >= 0) and (alpha <= 1) and (
                    np.max(np.abs(z - points[ind[1]] - alpha * (points[ind[0]] - points[ind[1]]))) <= tol):
                return True

        return False

    else:

        A = np.vstack((np.ones(3, dtype=np.float64), points.T - z.reshape(-1, 1)))

        d = np.empty(3, dtype=np.float64)

        d[0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
        d[1] = A[2, 0] * A[1, 2] - A[1, 0] * A[2, 2]
        d[2] = A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1]

        lm = d / (d[0] + d[1] + d[2])

        return np.all(lm >= 0) and np.all(lm <= 1)


def in_hull(z, points, tol=1e-8):
    """ Returns True or False depending on whether or not z is within
    the convex hull of points with the specified tolerance.
    """

    if np.all(np.min(points, axis=0) <= z + tol) and np.all(np.max(points, axis=0) >= z - tol):

        if points.shape[1] == 1:
            return (z[0] >= np.min(points) - tol) and (z[0] <= np.max(points) + tol)

        if len(points) == 1:

            return np.max(np.abs(z - points[0])) <= tol

        elif len(points) == 2:

            #             alpha = ((z-points[1]) @ (points[0]-points[1])) / ((points[0]-points[1]) @ (points[0]-points[1]))

            #             return (alpha >= 0) and (alpha <= 1) and (np.max(np.abs(z - points[1] - alpha*(points[0]-points[1]))) <= tol)

            return is_between_2D(z, points[0], points[1], tol=0.0)

        elif len(points) == 3:

            try:
                return in_triangle_2D(z, points, tol=0.0)

            except Exception as ex:

                print('points = ', points)
                print('z = ', z)
                raise ex

        else:

            ch = ConvexHull(points)

            return np.all(np.dot(ch.equations, np.r_[z, np.ones(1)]) <= tol)

    else:

        return False
