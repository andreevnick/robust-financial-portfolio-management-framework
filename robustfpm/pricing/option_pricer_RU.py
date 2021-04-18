# Copyright 2021 portfolio-robustfpm-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import gc
import numpy as np

from itertools import combinations
from scipy.spatial import ConvexHull
from sklearn.utils import check_random_state

from ..finance import IOption
from .set_handler import ISetHandler, RealSpaceHandler
from .lattice import Lattice
from ..util import coalesce, ProfilerData, Timer, PTimer, minksum_points, isin_points, minkprod_points
from robustfpm.cxhull import get_max_coordinates, in_hull


__all__ = ['OptionPricer']


class OptionPricer:
    '''
    Represents the numeric solver to the European option pricing problem under the robustfpm approach.
    
    Parameters
    ----------
    
    grid : class:`Grid`
        Point grid.
    N : int >= 0
        Investment horizon.
    option : class:`IOption`
        Option being priced.
    x0 : numpy.ndarray
        Vector of current prices.
    price_support : class:`ISetHandler`
        Set object representing the support K. Must be compact.
    price_dynamics : 'add' or 'mult'
        Additive or multiplicative price dynamics. Default is 'add'.
    constraint_set : class:`ISetHandler`
        Set object representing the trading constraints D.
    debug_mode: boolean
        If True, debug information is displayed during execution. Default is False.
    ignore_warnings: boolean
        If True, warnings from the linprog optimization procedures are not displayed. Default is False.
    enable_timer: boolean
        If True, profiler information will be displayed during execution. Default is False.
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
    Grid
    IOption
    ProfilerData
    :class:`robustfpm.pricing.problem.ConvhullSolver`
        An implementation for another class structure
        
    '''
    
    def __init__(self, grid, N, option, x0, price_support, price_dynamics='add', constraint_set=None,
                 debug_mode=False, ignore_warnings=False, enable_timer=False, profiler_data=None,
                 calc_market_strategies=False, pricer_options={}):
        
        self.grid = grid if isinstance(grid, Lattice) else Lattice(grid)
        self.n = grid.delta.size
        self.N = N
        
        assert isinstance(option, IOption), 'option must implement the IOption interface'
        self.option = option
        
        self.x0 = x0
        
        assert isinstance(price_support, ISetHandler), 'price_support must implement the ISetHandler interface'
        assert price_support.iscompact(), 'compactness of the price support is required'
        self.price_support = price_support
        
        assert price_dynamics in ['mult', 'add']
        self.price_dynamics = price_dynamics
        
        assert isinstance(constraint_set, ISetHandler), 'constraint_set must implement the ISetHandler interface'
        self.constraint_set = coalesce(constraint_set, RealSpaceHandler())
        
        self.debug_mode = debug_mode
        self.ignore_warnings = ignore_warnings
        self.enable_timer = enable_timer
        
        self.profiler_data = coalesce(profiler_data, ProfilerData())
        
        self.calc_market_strategies = calc_market_strategies
        
        if not isinstance(pricer_options, dict):
            pricer_options = {}
                    
        self.pricer_options = {
            'convex_hull_filter': pricer_options.get('convex_hull_filter', None),
            'convex_hull_prune_fail_count': pricer_options.get('convex_hull_prune_fail_count', 0),
            'convex_hull_prune_success_count': pricer_options.get('convex_hull_prune_success_count', 0),
            'convex_hull_prune_corner_n': pricer_options.get('convex_hull_prune_corner_n', 3),
            'convex_hull_prune_seed': pricer_options.get('convex_hull_prune_seed', None)
        }
        
        
    def __precalc(self):
        """ Init the required private attributes before the main pricing has started.
        
        """
        
        self.p0_ = self.grid.get_projection(self.x0) # map x0
        
#         self.dK_ = self.grid.get_projection(self.price_support) # map support neighbourhood
        
        self.silent_timer_ = not self.enable_timer
        
        self.pruning_random_state_ = check_random_state(self.pricer_options['convex_hull_prune_seed'])
        
        
    def get_support_set(self, curr_set, increment):
        """ Based on the set of price scenarios at K_{t-1} and the increment dK_t,
        returns the price scenarios at t.
        For additive dynamics, returns the set
            K_t = {x_{t-1} + y_t, x_{t-1} \in K_{t-1}, y_t \in dK_t};
        For multiplicative dunamics, returns the set
            K_t = {x_{t-1} * y_t, x_{t-1} \in K_{t-1}, y_t \in dK_t};
        
        """
        
        curr_set = np.atleast_2d(curr_set)
        
        assert isinstance(increment, ISetHandler), 'increment must implement the ISetHandler interface'
        
        if self.price_dynamics == 'add':
            
#             increment = np.atleast_2d(increment)
            
            return minksum_points(curr_set, increment.project(self.grid), recur_max_level=None)
        
        elif self.price_dynamics == 'mult':
            
#             assert isinstance(increment, ISetHandler), 'increment must implement the ISetHandler interface'
            
            return minkprod_points(self.grid, curr_set, increment, pos=True)
            
        else:
            
            raise ValueError('unknown price dynamics {dyn}'.format(dyn=self.price_dynamics))
        

    def generate_evaluation_point_lists(self, p0, dK, N, profiler_data=None):
        """ Precalculates the price scenarios for every t to avoid unnecessary evaluations of V_t.
        
        """

        with PTimer(header='V = [], V.append([p0])', silent=True, profiler_data=profiler_data):
            
            Vp = []
            Vp.append(p0.reshape(1,-1))
            
            Vf = [np.empty(Vp[-1].shape[0], dtype=Vp[-1].dtype)]

        for i in range(N):
            
            with PTimer(header='Vp.append(minksum_points(Vp[-1], dK, recur_max_level=None))', silent=True,
                        profiler_data=profiler_data):
                
#                 Vp.append(minksum_points(Vp[-1], dK, recur_max_level=None))
                Vp.append(self.get_support_set(Vp[-1], dK))
                Vf.append(np.empty(Vp[-1].shape[0], dtype=Vp[-1].dtype))

        return (Vp, Vf)
    
    
    def __chull_prune_points(self, xv):
        """ Pruning for the convex hull calculation. De facto not used.
        
        """
        
        fail_cnt = 0
        success_cnt = 0
        eps = 1e-8

        n = xv.shape[1]

        res_ind = np.arange(xv.shape[0])

        it = 0
        it_max = self.pricer_options['convex_hull_prune_fail_count'] * self.pricer_options['convex_hull_prune_success_count']
        
        while (xv.shape[0] > n) and (fail_cnt < self.pricer_options['convex_hull_prune_fail_count']) and (success_cnt < self.pricer_options['convex_hull_prune_success_count']):

            xv_size = xv.shape[0]

            ind_tf = np.ndarray((res_ind.shape[0], 2*n), dtype=np.bool)
        
            for i in range(n):
                ind_tf[:,2*i] = (xv[:,i] == np.amax(xv[:,i]))
                ind_tf[:,2*i+1] = (xv[:,i] == np.amin(xv[:,i]))

            ind = np.arange(xv.shape[0])[np.sum(ind_tf, axis=1) >= self.pricer_options['convex_hull_prune_corner_n']]

            if ind.shape[0] < n:
                print('few corner points')
                break

            ind_c = rs.choice(ind, size=n, replace=False)

            xc = np.vstack((np.ones(n, dtype=xv.dtype),
                           xv[ind_c,:-1].T))

            vc = xv[ind_c,-1]

            if np.linalg.matrix_rank(xc) != xc.shape[0]:
                fail_cnt += 1
#                 print('fail, rank')
        #         print('xc = ', xc)
        #         print('xv[ind] = ', xv[ind])
                continue


            ind_rest = np.arange(xv.shape[0])
            ind_rest = ind_rest[np.in1d(ind_rest, ind_c, assume_unique=True, invert=True)]

            x_rest = xv[ind_rest,:-1]
            v_rest = xv[ind_rest,-1]


            E = np.hstack((np.zeros((x_rest.shape[0],1)), x_rest))

            A = xc - E[...,np.newaxis]

            if n == 3:

                d12 = A[:,1,1] * A[:,2,2] - A[:,1,2] * A[:,2,1]
                d02 = A[:,2,0] * A[:,1,2] - A[:,1,0] * A[:,2,2]
                d01 = A[:,1,0] * A[:,2,1] - A[:,2,0] * A[:,1,1]

                detA = d12 + d02 + d01

                lmb = np.vstack( (d12, d02, d01) ).T / detA.reshape(-1,1)

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


        #     if (ind_remove.shape[0] > 0) and np.any(np.max(np.abs(xv[ind_remove] - np.array([[0.5, 0.9, 0.0]], dtype=xv.dtype)), axis=1) <= 0.001):
        #         print('x_rest, lmb, v, v_thresh')
        #         tmp = lmb @ vc
        #         for i in range(x_rest.shape[0]):
        #             print(x_rest[i,:], lmb[i,:], v_rest[i], tmp[i])

        #         print('xc = ', xc)
        #         print('vc = ', vc)
        #         print('xv[ind_remove] = ', xv[ind_remove])


            tf = np.in1d(np.arange(xv.shape[0]), ind_remove, assume_unique=True, invert=True)
            xv = xv[tf]
            res_ind = res_ind[tf]

        #     print('xv_size = ', xv_size)
        #     print('xv.shape[0] = ', xv.shape[0])

            it+=1
            if it > it_max:
                print('unexpected eternal loop')
                break
        

        return res_ind
    
    
    def __get_cxhull_value(self, x, v, z, calc_market_strategies, tol=1e-8):
        """ Returns the baricentric coordinates of base points x which
        correspond to the concave hull of {(x,v)} at z.
        
        """
        
        if self.pricer_options['convex_hull_filter'] is None:
            raise ValueError('convex_hull_filter is not specified')
            
            
        # short circuit for constant surface
        if np.abs(np.max(v) - np.min(v)) <= tol:
            
            ind = np.argmin(np.max(np.abs(x-z), axis=1))
            return (v[ind], [(ind,)])

        
        if len(x.shape) > 1:
            v = v.reshape(-1,1)
        
        points = np.hstack((x, v))
        
        try:
            pruned_ind = self.__chull_prune_points(points)
            points = points[res_ind]
            
        except:
            pruned_ind = np.arange(points.shape[0])
        
        
        points_zero = points[points[:,-1] > 0]
        points_zero[:,-1] = 0.0
        
        points = np.vstack((points,points_zero))
        
        if self.pricer_options['convex_hull_filter'] == 'qhull':
            
            ch = ConvexHull(points)
            
            if calc_market_strategies:            
            
                # find simplices whose projection contains z

                tf = [np.all(simplex < len(pruned_ind)) and in_hull(z, points[simplex,:-1], tol=tol)\
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
                    f[i] = get_max_coordinates(points[simplex][:,:-1], points[simplex][:,-1], z, tol=tol, debug_mode=self.debug_mode, ignore_warnings=self.ignore_warnings) @ points[simplex][:,-1]

                f = np.mean(f)


                # find the adversarial market strategies

                for dim in range(len(z)+1):

    #                 for ind in combinations(ch.vertices, dim+1):
    #                     if (max(ind) < len(pruned_ind)) and in_hull(z, points[ind,:-1])\
    #                     and (get_max_coordinates(points[ind, :-1], points[ind, -1], z, tol=tol, ignore_warnings=True) @ points[ind, -1] >= f - tol):
    #                         print('pruned_ind = ', pruned_ind)
    #                         print('pruned_ind.shape =', pruned_ind.shape)
    #                         print('ind = ', ind)

                    strategies = np.array([pruned_ind[list(ind)] for ind in combinations(ch.vertices, dim+1)\
                            if (max(ind) < len(pruned_ind)) and in_hull(z, points[list(ind),:-1])\
                            and (get_max_coordinates(points[list(ind), :-1], points[list(ind), -1], z, tol=tol, debug_mode=self.debug_mode, ignore_warnings=self.ignore_warnings) @ points[list(ind), -1] >= f - tol)
                           ])

                    if len(strategies) > 0:
                        break

                return (f, strategies)
            
            else:
                
#                 with Timer('Convex hull', flush=True):
                cv_point_indices = ch.vertices
#                 print('result = {0}/{1}'.format(cv_point_indices[cv_point_indices < x.shape[0]].shape[0], points.shape[0]))

#                 raise Exception('stopped')

#                 print('x.shape[0] = ', x.shape[0])
#                 print('cv_point_indices = ', cv_point_indices)
#                 print('pruned_ind', pruned_ind)

                res_ind = pruned_ind[cv_point_indices[cv_point_indices < len(pruned_ind)]]
    
                f = get_max_coordinates(x[res_ind], v[res_ind], z, debug_mode=self.debug_mode, ignore_warnings=self.ignore_warnings) @ v[res_ind]
    
                return (f, None)

        else:
        
            raise ValueError('unknown convex_hull_filter value \'{0}\''.format(self.pricer_options['convex_hull_filter']))


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
            

    def find_rho(self, x, v, K_x, convdK_x, calc_market_strategies):
        """ Returns the value function value V_t, given V_{t+1} at x_{t+1}.
        
        """
        
        convdK_x = np.atleast_2d(convdK_x)
        K_x      = np.atleast_2d(K_x)
        
#         supp_func = self.constraint_set.support_function(convdK_x - (1 if self.price_dynamics == 'mult' else 0))
        supp_func = self.constraint_set.support_function(convdK_x)
        
        tf = supp_func < np.Inf
        
        if np.sum(tf) == 0:
            
            print('support function is +Inf')
            return (-np.Inf, np.nan)
        
        K_x       = K_x[tf]
        convdK_x  = convdK_x[tf]
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

        return (res_u[maxind] - supp_func[maxind], convdK_x[maxind])
    
    
    def evaluate(self):
        """ Calculates the option value by backward-reconstructing the value function on a grid.
        Returns all the intermediate values of the value function Vf at points Vp. Vf[0][0] is
        the option value.
        
        """
        
        with PTimer(header='__precalc', silent=True, profiler_data=self.profiler_data) as tm:
            self.__precalc()
        
        with Timer('Основной расчет evaluate()', flush=False, silent=self.silent_timer_) as tm_total:

            pdata = self.profiler_data.data[tm_total.header]
            
            with Timer('Предрасчет точек, где будет вычисляться функция цены', silent=self.silent_timer_) as tm:
#                 Vp, Vf = self.generate_evaluation_point_lists(self.p0_, self.dK_, self.N, profiler_data=pdata.data[tm.header])
                Vp, Vf = self.generate_evaluation_point_lists(self.p0_, self.price_support, self.N, profiler_data=pdata.data[tm.header])


            with PTimer('Вычисление функции цены в конечный момент времени', silent=self.silent_timer_,
                      profiler_data=pdata) as tm:

                x = self.grid.map2x(Vp[-1])
                
#                 if self.price_dynamics == 'mult':
#                     x = np.exp(x)
                    
                Vf[-1] = self.option.payoff(x)

            with Timer('Вычисление функции цены в промежуточные моменты', silent=self.silent_timer_) as tm:
                
                pdata2 = pdata.data[tm.header]
                
                for t in reversed(range(self.N)):
                    
                    if not self.silent_timer_: print('t = {0}'.format(t))
                    
                    res = np.empty(Vp[t].shape[0], dtype=Vf[t+1].dtype)
                    
                    for i, vp in enumerate(Vp[t]):
                        
                        if not self.silent_timer_:
                            if (np.random.uniform()<0.001): print('iter = {0}/{1}'.format(i, len(Vp[t])))

                        with PTimer(header='K = vp + self.dK_', silent=True, profiler_data=pdata2) as tm2:
                            
#                             K = vp + self.dK_
                            K = self.get_support_set(vp, self.price_support)

                        with PTimer(header='tf = isin_points(Vp[t+1], K)', silent=True, profiler_data=pdata2) as tm2:
                            
                            tf = isin_points(Vp[t+1], K)

                        with PTimer(header='find_rho', silent=True, profiler_data=pdata2) as tm2:
                            
                            res_v, _ = self.find_rho(self.grid.map2x(Vp[t+1][tf]),
                                                     Vf[t+1][tf],
                                                     self.grid.map2x(K),
#                                                      self.grid.map2x(self.price_support.project(self.grid)),
                                                     self.grid.map2x(K-vp),
                                                     self.calc_market_strategies)
                            res[i] = res_v
                
#                             print('vp = ', self.grid.map2x(vp))
#                             print('res_v = ', res_v)
#                             print('Vp[t+1], Vf[t+1] = ')
#                             for c1, c2 in zip(self.grid.map2x(Vp[t+1][tf]), Vf[t+1][tf]):
#                                 print(c1, c2)

                    Vf[t] = res                
                
        gc.collect()

        return (Vf, Vp)