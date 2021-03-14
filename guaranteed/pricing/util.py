# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

""" Utility functions for pricing module"""
import numpy as np
from .multival_map import PIDynamics
from ..util import minkprod_points, minksum_points, unique_points_union, PTimer

__all__ = ['get_support_set',
           'generate_evaluation_point_lists']


def get_support_set(curr_set, price_dynamics, grid, t):
    """ Based on the set of price scenarios at :math:`K_{t-1}` and the increment :math:`dK_t`,
    returns the price scenarios at t.
    For additive dynamics, returns the set

    .. math :: K_t = {x_{t-1} + y_t, x_{t-1} \in K_{t-1}, y_t \in dK_t}

    For multiplicative dynamics, returns the set

    .. math :: K_t = {x_{t-1} * y_t, x_{t-1} \in K_{t-1}, y_t \in dK_t}

    """

    curr_set = np.atleast_2d(curr_set)

    price_dynamics_type = price_dynamics.type
    if isinstance(price_dynamics, PIDynamics):
        increment = price_dynamics(t=t)
        if price_dynamics_type == 'add':
            return minksum_points(curr_set, increment.project(grid), recur_max_level=None)
        else:
            return minkprod_points(grid, curr_set, increment, pos=True)
    else:
        if price_dynamics_type == 'add':
            res = np.array([])
            for pt in curr_set:
                increment = price_dynamics(x=grid.map2x(pt), t=t)
                # res = unique_points_union(res, minksum_points(pt, increment.project(grid), recur_max_level=None))
                res = unique_points_union(res, increment.add(grid.map2x(pt)).project(grid))
            return res
        else:
            res = np.array([])
            for pt in curr_set:
                increment = price_dynamics(x=grid.map2x(pt), t=t)
                res = unique_points_union(res, increment.multiply(grid.map2x(pt)).project(grid))
            return res


def generate_evaluation_point_lists(p0, grid, price_dynamics, N, profiler_data=None):
    """ Precalculates the price scenarios for every t to avoid unnecessary evaluations of V_t.

    """

    with PTimer(header='V = [], V.append([p0])', silent=True, profiler_data=profiler_data):
        Vp = [p0.reshape(1, -1)]

        Vf = [np.empty(Vp[-1].shape[0], dtype=Vp[-1].dtype)]

    for i in range(N):
        with PTimer(header='Vp.append(minksum_points(Vp[-1], dK, recur_max_level=None))', silent=True,
                    profiler_data=profiler_data):
            #                 Vp.append(minksum_points(Vp[-1], dK, recur_max_level=None))
            Vp.append(get_support_set(curr_set=Vp[-1], grid=grid, price_dynamics=price_dynamics, t=i + 1))
            Vf.append(np.empty(Vp[-1].shape[0], dtype=Vp[-1].dtype))

    return Vp, Vf
