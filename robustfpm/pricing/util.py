# Copyright 2021 portfolio-robustfpm-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

""" Utility functions for pricing module"""
import numpy as np
from .multival_map import PIDynamics, IdenticalMap
from .set_handler import EllipseHandler, RealSpaceHandler
from ..util import minkprod_points, minksum_points, unique_points_union, PTimer, square_neighbourhood_on_lattice
from warnings import warn

__all__ = ['get_support_set',
           'generate_evaluation_point_lists',
           'get_a_star',
           'get_constraints_lipschitz']


def check_point_in_set(set, value, tval, xval=None):
    r"""
    Check that point with all coordinates equal to `value` is in `increment`, raise warning if `False`.

    Returns True if :math:`x = (v,v,\dots,v) \in int(\mathcal{X})`

    Parameters
    ----------
    set : ISetHandler
        Set :math:`\mathcal{X}`
    value : numerical
        Coordinate value :math:`v`
    tval : int
        Time point, for warning only
    xval :
        Coordinate of x, for warning only

    Returns
    -------
    bool
    """
    msg = 'For t={tval}, zero is not in additive increment!'.format(tval=tval) \
        if xval is None else 'For t={tval}, x={xval} zero is not in additive increment!'.format(tval=tval, xval=xval)
    if not set.is_interior(np.full(set.dim, value)):
        warn(msg)
        return False
    return True


def get_r_cur(increment, r_star):
    r"""
    Get current value of :math:`r^*_t`

    Parameters
    ----------
    increment : ISetHandler
        Additive increment
    r_star : np.float
        Previous value of :math:`r^*_t`


    Returns
    -------
    np.float
        Maximum of current and previous values for :math:`r^*_t`

    """
    increment_bounding_box = increment.boundaries()
    r_cur = np.min(increment_bounding_box[:, 1] - increment_bounding_box[:, 0]) / 2
    return r_cur if r_star >= r_cur else r_star


def get_support_set(curr_set, price_dynamics, lattice, t, calc_r_star=False):
    """
    Return possible price scenarios.

    Based on the set of price scenarios at :math:`K_{t-1}` and the increment :math:`dK_t`,
    returns the price scenarios at t.
    For additive dynamics, returns the set

    .. math :: K_t = {x_{t-1} + y_t, x_{t-1} \in K_{t-1}, y_t \in dK_t}

    For multiplicative dynamics, returns the set

    .. math :: K_t = {x_{t-1} \cdot y_t, x_{t-1} \in K_{t-1}, y_t \in dK_t}

    Returns
    -------
    res : np.ndarray
        Resulting support set :math:`K_t`
    r_star : np.float64, optional
        :math:`r^*_t` for `price_dynamics`. That is, smallest possible inner radius of additive increment. Used in precision calculation.
    """

    curr_set = np.atleast_2d(curr_set)

    price_dynamics_type = price_dynamics.type
    r_star = np.inf
    if isinstance(price_dynamics, PIDynamics):
        increment = price_dynamics(t=t)
        if price_dynamics_type == 'add':
            increment_projection = increment.project(lattice)
            if calc_r_star and check_point_in_set(increment, 0, t):
                increment_bounding_box = increment.boundaries()
                r_star = np.min(increment_bounding_box[:, 1] - increment_bounding_box[:, 0]) / 2
            res = minksum_points(curr_set, increment_projection, recur_max_level=None)
        else:
            res = minkprod_points(lattice, curr_set, increment, pos=True)
            if calc_r_star and check_point_in_set(increment, 1, t):
                for pt in curr_set:
                    x = lattice.map2x(pt)
                    add_inc = increment.multiply(x).add(-x)
                    r_star = get_r_cur(add_inc, r_star)

    else:
        curr_set = minksum_points(curr_set, square_neighbourhood_on_lattice(np.zeros_like(lattice.delta), 1, True))
        if price_dynamics_type == 'add':
            res = None
            for pt in curr_set:
                increment = price_dynamics(x=lattice.map2x(pt), t=t)
                if calc_r_star and check_point_in_set(increment, 0, t):
                    r_star = get_r_cur(increment, r_star)
                res = increment.add(lattice.map2x(pt)).project(lattice) if res is None \
                    else unique_points_union(res, increment.add(lattice.map2x(pt)).project(lattice))
        else:
            res = None
            for pt in curr_set:
                x = lattice.map2x(pt)
                increment = price_dynamics(x=x, t=t)
                if calc_r_star and check_point_in_set(increment, 1, t):
                    add_inc = increment.multiply(x).add(-x)
                    r_star = get_r_cur(add_inc, r_star)
                res = increment.multiply(lattice.map2x(pt)).project(lattice) if res is None \
                    else unique_points_union(res, increment.multiply(lattice.map2x(pt)).project(lattice))
    if calc_r_star:
        return res, r_star
    return res


def generate_evaluation_point_lists(p0, lattice, price_dynamics, N, profiler_data=None, calc_r_star=False):
    r"""
    Precalculate all price scenarios and calculate some parameters for precision estimation.

    Precalculates the price scenarios for every `t` to avoid unnecessary evaluations of :math:`V_t`.

    If `calc_r_star` is True, also returns an array of :math:`r^*_t` — smallest radiuses of circles, inscribed into additive increments (for precision calculation).
    """

    with PTimer(header='V = [], V.append([p0]), r_star = []', silent=True, profiler_data=profiler_data):
        Vp = [p0.reshape(1, -1)]
        Vf = [np.empty(Vp[-1].shape[0], dtype=Vp[-1].dtype)]
        r_star = []

    for i in range(N):
        with PTimer(header='Vp.append(minksum_points(Vp[-1], dK, recur_max_level=None))', silent=True,
                    profiler_data=profiler_data):
            #                 Vp.append(minksum_points(Vp[-1], dK, recur_max_level=None))
            if calc_r_star:
                cur_Vp, cur_r = get_support_set(curr_set=Vp[-1], lattice=lattice, price_dynamics=price_dynamics,
                                                t=i + 1,
                                                calc_r_star=True)
            else:
                cur_Vp = get_support_set(curr_set=Vp[-1], lattice=lattice, price_dynamics=price_dynamics, t=i + 1,
                                         calc_r_star=False)
            Vp.append(cur_Vp)
            Vf.append(np.empty(Vp[-1].shape[0], dtype=Vp[-1].dtype))
            if calc_r_star:
                r_star.append(cur_r)
    if calc_r_star:
        return Vp, Vf, r_star
    return Vp, Vf


def get_a_star(option, lattice, r_star, max_time, price_dynamics, trading_constraints, points, eps=None):
    r"""
    Calculate :math:`A^*` with given precision

    Parameters
    ----------
    option : IOption
    lattice : Lattice
    r_star : np.array
    max_time : int
    price_dynamics : PriceDynamics
    trading_constraints : IMultivalMap
    points : np.ndarray
        Precalculated points `Vp`
    eps : np.float, optional
        Precision for :math:`A^*` calculation, defaults to :code:`min(lattice.delta)`

    Returns
    -------
    A_max : np.ndarray
        Values for :math:`A^*_t`
    """

    def is_acceptable_radius(r, t):
        dim = price_dynamics.dim
        B = EllipseHandler(np.full(dim, 0), np.diag([r ** 2 for i in range(dim)]))
        B = lattice.map2x(B.project(lattice))
        idx_border = np.apply_along_axis(np.linalg.norm, 1, B) >= r - np.linalg.norm(lattice.delta)
        S = B[idx_border]
        min_sigmaK = np.inf
        for pt in points[t]:
            x = lattice.map2x(pt)
            cur_D = trading_constraints(x, t)
            in_d_idx = cur_D.contains(S)
            K = price_dynamics(x, t).multiply(x).add(-x) if price_dynamics.type == 'mult' else price_dynamics(t, x)
            min_sigmaK = np.minimum(min_sigmaK,
                                    np.min(K.support_function(-S[in_d_idx]))) if np.any(in_d_idx) else min_sigmaK
        return min_sigmaK <= c_star[t - 1]

    eps = eps if eps is not None else np.min(lattice.delta)
    c = [np.max(option.payoff(prices=lattice.map2x(points[t + 1]), t=t + 1)) for t in range(max_time)]
    c = np.array(c)
    c_star = np.flip(np.maximum.accumulate(np.flip(c)))
    A_max = c_star / r_star
    if isinstance(trading_constraints, IdenticalMap) and isinstance(trading_constraints.support, RealSpaceHandler):
        return A_max
    # else – we begin iterating
    A_min = np.full_like(A_max, 0)
    for t in range(max_time):
        while A_max[t] - A_min[t] >= eps:
            A_new = (A_max[t] + A_min[t]) / 2
            if is_acceptable_radius(A_new, t + 1):
                A_min[t] = A_new
            else:
                A_max[t] = A_new
    A_max = (A_max + A_min) / 2
    return A_max


def get_constraints_lipschitz(max_time, points, lattice, trading_constraints):
    r"""
    Get Lipschitz constants for :math:`\sigma_{D_t(\cdot)}`

    For each `t`, returns :math:`L_{D_t} = \max\limits_{x \in B_t(\cdot)} L_{\sigma_{D_t(x)}}`

    Parameters
    ----------
    max_time : int
    points : list
    lattice : Lattice
    trading_constraints : IMultivalMap

    Returns
    -------

    L_D : np.ndarray
        Corresponding values of `:math:`L_{D_t}`

    Notes
    -----

    If :math:`D_t(x)` is unbounded for `all` `x`, corresponding :math:`L_{D_t}` is set to zero.
    This is the case for :math:`D_t(\cdot) = \mathbb{R}^n` and :math:`D_t(\cdot) = \mathbb{R}^n_+`.

    However, this is not *always* the case.
    """
    L_D = np.full(max_time, 0)  # todo: set another default value? throw errors if undefined?
    if isinstance(trading_constraints, IdenticalMap):
        cur_D = trading_constraints()
        if cur_D.iscompact():
            L_D = [np.max(np.apply_along_axis(np.linalg.norm, 1, lattice.map2x(cur_D.project(lattice)))) for t in
                   range(max_time)]
    else:
        for t in range(max_time):
            for pt in points[t]:
                x = lattice.map2x(pt)
                cur_D = trading_constraints(x=x, t=t)
                if cur_D.iscompact():
                    L_D[t] = np.maximum(L_D[t], np.max(
                        np.apply_along_axis(np.linalg.norm, 1, lattice.map2x(cur_D.project(lattice)))))
    return L_D
