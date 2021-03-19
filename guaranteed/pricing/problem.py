# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

""" This submodule implements a :class:`Problem` class that
incapsulates a problem of pricing and hedging an option via a guaranteed approach
and solver classes for solving such problems: :class:`ISolver` as an interface and some realisations."""

import gc
import numpy as np

from abc import ABC, abstractmethod

from itertools import combinations
from scipy.spatial import ConvexHull
from sklearn.utils import check_random_state

from .multival_map import PriceDynamics, IMultivalMap
from ..finance import IOption, AmericanOption
from ..pricing import Lattice
from ..util import Timer, PTimer, ProfilerData, coalesce, isin_points
from .util import get_support_set, generate_evaluation_point_lists
from ..cxhull import get_max_coordinates, in_hull

__all__ = ['Problem',
           'ISolver',
           'ConvhullSolver']


class Problem:
    r""" A class representing a problem of pricing and hedging an option via a guaranteed approach.

    TODO: Provide detailed explanation of a problem here

    Parameters
    ----------
    starting_price: np.ndarray, size = (n,)
        Starting prices of assets, as points from :math:`\mathbb{R}^{n}`
    price_dynamics: PriceDynamics
        Price dynamics of a problem. Must be defined for all t <= time_horizon.
    trading_constraints: IMultivalMap
        A mapping :math:`D_t(\cdot)`, representing trading constraints
    option: IOption
        An option to price (and hedge)
    lattice: Lattice
        A lattice to solve problem on
    time_horizon: int, optional
        Time horizon :math:`N`. If not given, then deduced from `option` expiration and `price_dynamics` (if possible).
    solver: ISolver
        Solver for a problem

    """

    def __init__(self, starting_price, price_dynamics, trading_constraints, option,
                 lattice, time_horizon=None, solver=None):
        self.starting_price = np.array(starting_price).reshape((1, -1))
        if not isinstance(price_dynamics, PriceDynamics):
            raise TypeError('Price dynamics must be an instance of PriceDynamics!')
        self.price_dynamics = price_dynamics
        if not isinstance(trading_constraints, IMultivalMap):
            raise TypeError('Trading constrainst must be a multivalued mapping')
        self.trading_constraints = trading_constraints
        if not isinstance(option, IOption):
            raise TypeError('Option must implement IOption interface!')
        self.option = option
        self.time_horizon = time_horizon
        # if no time_horizon is provided, try to take it from an option
        if time_horizon is None:
            if not np.isinf(option.expiry):
                self.time_horizon = option.expiry
            elif self.price_dynamics.t_max > 0 and not (np.isinf(self.price_dynamics.t_max)):
                self.time_horizon = self.price_dynamics.t_max
            else:
                raise TypeError('Can not determine time horizon! Problem is unbounded.')
        if self.time_horizon <= 0:
            raise ValueError('Wrong time horizon! Must be a positive integer.')
        # if the option is american, set its expiration to time_horizon
        if not isinstance(self.option, AmericanOption) and self.option.expiry != self.time_horizon:
            raise ValueError(
                'Expiration of an option ({exp_opt}) and given time horizon ({th}) do not match!'.format(
                    exp_opt=option.expiry, th=self.time_horizon
                ))
        # check that price_dynamics is defined for all time
        if self.time_horizon > self.price_dynamics.t_max:
            raise ValueError('Price dynamics must be defined for all time t <= time_horizon!')

        self.lattice = lattice
        self.dim = lattice.delta.shape[0]
        # check dimensions
        if (not np.isinf(self.price_dynamics.dim) and self.price_dynamics.dim != self.dim) or (
                not np.isinf(self.trading_constraints.dim) and self.trading_constraints.dim != self.dim) or (
                self.starting_price.shape[1] != self.dim):
            raise ValueError('Dimensions of price_dynamics, trading_constraints, starting_price and lattice must match!')

        self.solver = solver
        self.hedge = None
        self.Vp = None
        self.Vx = None

    def get_precision(self):
        """ Calculates the precision of given problem

        Returns
        -------
        float
            Precision for value function
        """
        # TODO: calculate and return precision via lipshitz constants of price_dynamics and trading_constraints
        raise NotImplementedError('This method is not yet implemented')

    def set_precision(self, precision):
        """ Changes lattice to match given precision

        Parameters
        ----------
        precision: float
            target precision
        """
        # TODO: implement
        raise NotImplementedError('This method is not yet implemented')

    def solve(self, calc_hedge=False):
        """ Solves a problem and sets its Vp and Vx (and hedge, if `calc_hedge` is True) to solution

        Parameters
        ----------
        calc_hedge: bool, optional, default = False
            If True, also calculates hedging strategy (might be sufficiently slower)
        """

        if self.solver is None:
            raise ValueError('Solver is not set! Can not solve a problem')
        solution = self.solver.solve(self, calc_hedge)
        if calc_hedge:
            self.Vx, self.Vp, self.hedge = solution
        else:
            self.Vx, self.Vp = solution


class ISolver(ABC):

    @abstractmethod
    def solve(self, problem, calc_hedge=False):
        """ Solves a given problem and calculates hedging strategy (if `calc_hedge` is True)

        Parameters
        ----------
        problem: Problem
            Problem to solve
        calc_hedge: bool, optional, default = False
            If True, also tries to return hedging strategy

        Returns
        -------
        tuple
            First element is values of value function Vx, second is points on `problem.lattice` on which Vx is calculated,
             third (optional) is hedging strategy.
        """
        raise NotImplementedError('This method should be implemented!')


# noinspection PyPep8Naming,PyBroadException
class ConvhullSolver(ISolver):
    """
    Represents the numeric solver to the option pricing problem under the guaranteed approach.

    TODO: write detailed description

    Parameters
    ----------
    debug_mode: boolean
        If True, debug information is displayed during execution. Default is False.
    ignore_warnings: boolean
        If True, warnings from the linprog optimization procedures are not displayed. Default is False.
    enable_timer: boolean
        If True, profiler information will be displayed during execution. Default is False.
    iter_tick: int
        If `enable_timer` is True, then timer will tick, on average, each `iter_tick` iteration. Default is 1000.
    profiler_data: class:'ProfilerData'
        Profiler data, to which the execution timing can be appended to. If None, a new profiler data
        object will be created. Default is None.
    calc_market_strategies: boolean
        If True, adverse market strategies at every step will calculated. Not used for pricing.
        True leads to the slower execution speed. Default is False.
    pricer_options: dict
        Options for numerical methods.

    See also
    --------
    :class:`guaranteed.pricing.option_pricer_RU.OptionPricer`

    """

    def solve(self, problem, calc_hedge=False):
        self.calc_market_strategies = calc_hedge
        return self.evaluate(x0=problem.starting_price, lattice=problem.lattice, price_dynamics=problem.price_dynamics,
                             trading_constraints=problem.trading_constraints, max_time=problem.time_horizon,
                             option=problem.option)

    def __init__(self, debug_mode=False, ignore_warnings=False, enable_timer=False, iter_tick=1000, profiler_data=None,
                 calc_market_strategies=False, pricer_options=None):

        self.debug_mode = debug_mode
        self.ignore_warnings = ignore_warnings
        self.enable_timer = enable_timer
        self.iter_tick = 1 / iter_tick

        self.profiler_data = coalesce(profiler_data, ProfilerData())

        self.calc_market_strategies = calc_market_strategies

        if not isinstance(pricer_options, dict):
            pricer_options = {}

        self.pricer_options = {
            'convex_hull_filter': pricer_options.get('convex_hull_filter', 'qhull'),
            'convex_hull_prune_fail_count': pricer_options.get('convex_hull_prune_fail_count', 0),
            'convex_hull_prune_success_count': pricer_options.get('convex_hull_prune_success_count', 0),
            'convex_hull_prune_corner_n': pricer_options.get('convex_hull_prune_corner_n', 3),
            'convex_hull_prune_seed': pricer_options.get('convex_hull_prune_seed', None)
        }

    def __precalc(self, x0, lattice):
        """ Init the required private attributes before the main pricing has started.

        """

        self.p0_ = lattice.get_projection(x0)  # map x0

        self.silent_timer_ = not self.enable_timer

        self.pruning_random_state_ = check_random_state(self.pricer_options['convex_hull_prune_seed'])

    # noinspection PyPep8Naming
    def __chull_prune_points(self, xv):
        """ Pruning for the convex hull calculation. De facto not used.

        """

        fail_cnt = 0
        success_cnt = 0
        eps = 1e-8

        n = xv.shape[1]

        res_ind = np.arange(xv.shape[0])

        it = 0
        it_max = self.pricer_options['convex_hull_prune_fail_count'] * self.pricer_options[
            'convex_hull_prune_success_count']

        while (xv.shape[0] > n) and (fail_cnt < self.pricer_options['convex_hull_prune_fail_count']) and (
                success_cnt < self.pricer_options['convex_hull_prune_success_count']):

            xv_size = xv.shape[0]

            ind_tf = np.ndarray((res_ind.shape[0], 2 * n), dtype=np.bool)

            for i in range(n):
                ind_tf[:, 2 * i] = (xv[:, i] == np.amax(xv[:, i]))
                ind_tf[:, 2 * i + 1] = (xv[:, i] == np.amin(xv[:, i]))

            ind = np.arange(xv.shape[0])[np.sum(ind_tf, axis=1) >= self.pricer_options['convex_hull_prune_corner_n']]

            if ind.shape[0] < n:
                print('few corner points')
                break

            ind_c = np.random.choice(ind, size=n, replace=False)

            xc = np.vstack((np.ones(n, dtype=xv.dtype),
                            xv[ind_c, :-1].T))

            vc = xv[ind_c, -1]

            if np.linalg.matrix_rank(xc) != xc.shape[0]:
                fail_cnt += 1
                #                 print('fail, rank')
                #         print('xc = ', xc)
                #         print('xv[ind] = ', xv[ind])
                continue

            ind_rest = np.arange(xv.shape[0])
            ind_rest = ind_rest[np.in1d(ind_rest, ind_c, assume_unique=True, invert=True)]

            x_rest = xv[ind_rest, :-1]
            v_rest = xv[ind_rest, -1]

            E = np.hstack((np.zeros((x_rest.shape[0], 1)), x_rest))

            A = xc - E[..., np.newaxis]

            if n == 3:

                d12 = A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1]
                d02 = A[:, 2, 0] * A[:, 1, 2] - A[:, 1, 0] * A[:, 2, 2]
                d01 = A[:, 1, 0] * A[:, 2, 1] - A[:, 2, 0] * A[:, 1, 1]

                detA = d12 + d02 + d01

                lmb = np.vstack((d12, d02, d01)).T / detA.reshape(-1, 1)

            else:
                raise ValueError('n <> 3 is not supported')

            ind_remove = ind_rest[np.bitwise_and(np.all(lmb >= 0, axis=1), v_rest <= lmb @ vc + eps)]

            if ind_remove.shape[0] == 0:
                #                 print('fail, not found')
                #                 print('xv[ind_c] = ', xv[ind_c])
                fail_cnt += 1
            else:
                #                 print('success')
                success_cnt += 1
                fail_cnt = 0

            # if (ind_remove.shape[0] > 0) and np.any(np.max(np.abs(xv[ind_remove] - np.array([[0.5, 0.9, 0.0]],
            # dtype=xv.dtype)), axis=1) <= 0.001): print('x_rest, lmb, v, v_thresh') tmp = lmb @ vc for i in range(
            # x_rest.shape[0]): print(x_rest[i,:], lmb[i,:], v_rest[i], tmp[i])

            #         print('xc = ', xc)
            #         print('vc = ', vc)
            #         print('xv[ind_remove] = ', xv[ind_remove])

            tf = np.in1d(np.arange(xv.shape[0]), ind_remove, assume_unique=True, invert=True)
            xv = xv[tf]
            res_ind = res_ind[tf]

            #     print('xv_size = ', xv_size)
            #     print('xv.shape[0] = ', xv.shape[0])

            it += 1
            if it > it_max:
                print('unexpected eternal loop')
                break

        return res_ind

    # noinspection PyUnboundLocalVariable
    def __get_cxhull_value(self, x, v, z, calc_market_strategies, tol=1e-8):
        """ Returns the baricentric coordinates of base points x which
        correspond to the concave hull of {(x,v)} at z.

        """

        if self.pricer_options['convex_hull_filter'] is None:
            raise ValueError('convex_hull_filter is not specified')

        # short circuit for constant surface
        if np.abs(np.max(v) - np.min(v)) <= tol:
            ind = np.argmin(np.max(np.abs(x - z), axis=1))
            return v[ind], [(ind,)]

        if len(x.shape) > 1:
            v = v.reshape(-1, 1)

        points = np.hstack((x, v))

        try:
            pruned_ind = self.__chull_prune_points(points)
            points = points[pruned_ind]

        except:
            pruned_ind = np.arange(points.shape[0])

        points_zero = points[points[:, -1] > 0]
        points_zero[:, -1] = 0.0

        points = np.vstack((points, points_zero))

        if self.pricer_options['convex_hull_filter'] == 'qhull':

            ch = ConvexHull(points)

            if calc_market_strategies:

                # find simplices whose projection contains z

                tf = [np.all(simplex < len(pruned_ind)) and in_hull(z, points[simplex, :-1], tol=tol) \
                      for i, simplex in enumerate(ch.simplices)]

                #                 plt.figure(figsize=(10,10))
                #                 plt.scatter(points[pruned_ind,0], points[pruned_ind,1])
                #                 plt.show()

                #                 for i, simplex in enumerate(ch.simplices):
                #                     if np.all(simplex < len(pruned_ind)):
                #                         print('--- {0} ---'.format(i))
                #                         print('points[simplex,:-1] = ', points[simplex,:-1])
                #                         print('z = ', z)

                opt_simplices = ch.simplices[tf]

                # find the convex hull value at z

                f = np.empty(len(opt_simplices), dtype=np.float64)

                for i, simplex in enumerate(opt_simplices):
                    f[i] = get_max_coordinates(points[simplex][:, :-1], points[simplex][:, -1], z, tol=tol,
                                               debug_mode=self.debug_mode, ignore_warnings=self.ignore_warnings) @ \
                           points[simplex][:, -1]

                f = np.mean(f)

                # find the adversarial market strategies

                for dim in range(len(z) + 1):

                    # for ind in combinations(ch.vertices, dim+1): if (max(ind) < len(pruned_ind)) and in_hull(z,
                    # points[ind,:-1])\ and (get_max_coordinates(points[ind, :-1], points[ind, -1], z, tol=tol,
                    # ignore_warnings=True) @ points[ind, -1] >= f - tol): print('pruned_ind = ', pruned_ind) print(
                    # 'pruned_ind.shape =', pruned_ind.shape) print('ind = ', ind)

                    strategies = np.array([pruned_ind[list(ind)] for ind in combinations(ch.vertices, dim + 1) \
                                           if (max(ind) < len(pruned_ind)) and in_hull(z, points[list(ind), :-1]) \
                                           and (get_max_coordinates(points[list(ind), :-1], points[list(ind), -1], z,
                                                                    tol=tol, debug_mode=self.debug_mode,
                                                                    ignore_warnings=self.ignore_warnings) @ points[
                                                    list(ind), -1] >= f - tol)
                                           ])

                    if len(strategies) > 0:
                        break

                return f, strategies

            else:

                #                 with Timer('Convex hull', flush=True):
                cv_point_indices = ch.vertices
                # print('result = {0}/{1}'.format(cv_point_indices[cv_point_indices < x.shape[0]].shape[0],
                # points.shape[0]))

                #                 raise Exception('stopped')

                #                 print('x.shape[0] = ', x.shape[0])
                #                 print('cv_point_indices = ', cv_point_indices)
                #                 print('pruned_ind', pruned_ind)

                res_ind = pruned_ind[cv_point_indices[cv_point_indices < len(pruned_ind)]]

                f = get_max_coordinates(x[res_ind], v[res_ind], z, debug_mode=self.debug_mode,
                                        ignore_warnings=self.ignore_warnings) @ v[res_ind]

                return f, None

        else:

            raise ValueError(
                'unknown convex_hull_filter value \'{0}\''.format(self.pricer_options['convex_hull_filter']))

    def find_u(self, x, v, z, calc_market_strategies):
        """ Returns u(z), see the algorithm.

        """

        try:
            Vopt, strategies_ind = self.__get_cxhull_value(x, v, z, calc_market_strategies, 1e-10)

        except Exception as ex:

            print('x = ', x)
            print('v = ', v)
            print('z = ', z)

            raise ex

        # calculate probabilities for the market strategies via get_max_coordinates
        # NOT REQUIRED IN THIS VERSION

        #         strategies = [(x[ind], get_max_coordinates(x[ind], v[ind], z), v[ind]) for ind in strategies_ind]

        return Vopt

    def find_rho(self, x, v, K_x, convdK_x, calc_market_strategies, constraint_set):
        """ Returns the value function value V_t, given V_{t+1} at x_{t+1}.

        """

        convdK_x = np.atleast_2d(convdK_x)
        K_x = np.atleast_2d(K_x)

        supp_func = constraint_set.support_function(convdK_x)

        tf = supp_func < np.Inf

        if np.sum(tf) == 0:
            print('support function is +Inf')
            return -np.Inf, np.nan

        K_x = K_x[tf]
        convdK_x = convdK_x[tf]
        supp_func = supp_func[tf]

        n = x.shape[1]

        res_u = np.ndarray(K_x.shape[0], dtype=v.dtype)

        for i in range(K_x.shape[0]):
            Vopt = self.find_u(x, v, K_x[i], calc_market_strategies)

            res_u[i] = Vopt

        #         for c in zip(convdK_x, res_u, supp_func, res_u - supp_func):
        #             print('convdK_x, res_u, supp_func, res_u - supp_func: ', c)

        maxind = np.argmax(res_u - supp_func)
        #         print('maxind = ', maxind)

        return res_u[maxind] - supp_func[maxind], convdK_x[maxind]

    def evaluate(self, x0, lattice, price_dynamics, trading_constraints, max_time, option):
        """ Calculates the option value by backward-reconstructing the value function on a lattice.
        Returns all the intermediate values of the value function Vf at points Vp.
        Vf[0][0] is the option value.

        """

        with PTimer(header='__precalc', silent=True, profiler_data=self.profiler_data) as tm:
            self.__precalc(x0=x0, lattice=lattice)

        with Timer('Solving the problem', flush=False, silent=self.silent_timer_) as tm_total:

            pdata = self.profiler_data.data[tm_total.header]

            with Timer('Precalculating points for value function evaluation', silent=self.silent_timer_) as tm:
                Vp, Vf = generate_evaluation_point_lists(p0=self.p0_, lattice=lattice, price_dynamics=price_dynamics,
                                                         N=max_time, profiler_data=pdata.data[tm.header])
            # price check (negative prices)
            if np.any([np.any(lattice.map2x(Vp[i]) < 0) for i in range(max_time)]):
                raise TypeError('Prices can become negative! The problem is badly stated.')

            with PTimer('Computing value function in the last point', silent=self.silent_timer_,
                        profiler_data=pdata) as tm:

                x = lattice.map2x(Vp[-1])
                Vf[-1] = option.payoff(x, max_time)

            with Timer('Computing value function in intermediate points in time', silent=self.silent_timer_) as tm:

                pdata2 = pdata.data[tm.header]

                for t in reversed(range(max_time)):

                    if not self.silent_timer_:
                        print('t = {0}'.format(t))

                    res = np.empty(Vp[t].shape[0], dtype=Vf[t + 1].dtype)

                    for i, vp in enumerate(Vp[t]):

                        if not self.silent_timer_:
                            if np.random.uniform() < self.iter_tick:
                                print('iter = {0}/{1} ({2:.2f}%)'.format(i, len(Vp[t]), 100 * i / len(Vp[t])))

                        with PTimer(header='K = vp + self.dK_', silent=True, profiler_data=pdata2) as tm2:

                            #                             K = vp + self.dK_
                            K = get_support_set(vp, lattice=lattice, price_dynamics=price_dynamics, t=t)

                        with PTimer(header='tf = isin_points(Vp[t+1], K)', silent=True, profiler_data=pdata2) as tm2:

                            tf = isin_points(Vp[t + 1], K)

                        with PTimer(header='find_rho', silent=True, profiler_data=pdata2) as tm2:

                            res_v, _ = self.find_rho(lattice.map2x(Vp[t + 1][tf]),
                                                     Vf[t + 1][tf],
                                                     lattice.map2x(K),
                                                     lattice.map2x(K - vp),
                                                     self.calc_market_strategies,
                                                     constraint_set=trading_constraints(x=lattice.map2x(vp), t=t))
                            res[i] = res_v

                    #                             print('vp = ', self.lattice.map2x(vp))
                    #                             print('res_v = ', res_v)
                    #                             print('Vp[t+1], Vf[t+1] = ')
                    #                             for c1, c2 in zip(self.lattice.map2x(Vp[t+1][tf]), Vf[t+1][tf]):
                    #                                 print(c1, c2)

                    Vf[t] = np.maximum(res, option.payoff(prices=lattice.map2x(Vp[t]), t=t))

        gc.collect()

        return Vf, Vp
