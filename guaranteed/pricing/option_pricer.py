# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import gc
import numpy as np

from scipy.spatial import ConvexHull
from sklearn.utils import check_random_state

from ..finance import IOption
from .set_handler import ISetHandler
from .lattice import Lattice
from ..util import coalesce, ProfilerData, Timer, PTimer, minksum_points, isin_points
from guaranteed.cxhull import get_max_coordinates


__all__ = ['OptionPricer']


class OptionPricer:
    
    def __init__(self, lattice, N, option, x0, price_support, constraint_set=None,
                 debug_mode=False, ignore_warnings=False, enable_timer=False, profiler_data=None,
                 pricer_options={}):
        
        self.lattice = lattice if isinstance(lattice, Lattice) else Lattice(lattice)
        self.n = lattice.delta.size
        self.N = N
        
        assert isinstance(option, IOption), 'option must implement the IOption interface'
        self.option = option
        
        self.x0 = x0
        
        assert isinstance(price_support, ISetHandler), 'price_support must implement the ISetHandler interface'
        assert price_support.iscompact(), 'compactness of the price support is required'
        self.price_support = price_support
        
        assert isinstance(constraint_set, ISetHandler), 'constraint_set must implement the ISetHandler interface'
        self.constraint_set = coalesce(constraint_set, SetHandler('unbounded', verify_compactness=False))
        
        self.debug_mode = debug_mode
        self.ignore_warnings = ignore_warnings
        self.enable_timer = enable_timer
        
        self.profiler_data = coalesce(profiler_data, ProfilerData())
        
        if not isinstance(pricer_options, dict):
            pricer_options = {}
                    
        self.pricer_options = {
            'convex_hull_filter': pricer_options.get('convex_hull_filter', None),
            'convex_hull_prune_fail_count': pricer_options.get('convex_hull_prune_fail_count', 20),
            'convex_hull_prune_success_count': pricer_options.get('convex_hull_prune_success_count', 20),
            'convex_hull_prune_corner_n': pricer_options.get('convex_hull_prune_corner_n', 3),
            'convex_hull_prune_seed': pricer_options.get('convex_hull_prune_seed', None)
        }
        
        
    def __precalc(self):
        
        self.p0_ = self.lattice.get_projection(self.x0) # map x0
        
        self.dK_ = self.lattice.get_projection(self.price_support) # map support neighbourhood
        
        self.silent_timer_ = not self.enable_timer
        
        self.pruning_random_state_ = check_random_state(self.pricer_options['convex_hull_prune_seed'])
        

    def generate_evaluation_point_lists(self, p0, dK, N, profiler_data=None):

        with PTimer(header='V = [], V.append([p0])', silent=True, profiler_data=profiler_data):
            
            Vp = []
            Vp.append(p0.reshape(1,-1))
            
            Vf = [np.empty(Vp[-1].shape[0], dtype=Vp[-1].dtype)]

        for i in range(N):
            
            with PTimer(header='Vp.append(minksum_points(Vp[-1], dK, recur_max_level=None))', silent=True,
                        profiler_data=profiler_data):
                
                Vp.append(minksum_points(Vp[-1], dK, recur_max_level=None))
                Vf.append(np.empty(Vp[-1].shape[0], dtype=Vp[-1].dtype))

        return (Vp, Vf)
    
    
    def __chull_prune_points(self, xv):
        
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

    
    def __get_cvhull_indices(self, x, v):
        
        if self.pricer_options['convex_hull_filter'] is None:
            return np.arange(x.shape[0])

        if len(x.shape) > 1:
            v = v.reshape(-1,1)
        
        points = np.hstack((x, v))
        
        try:
            pruned_ind = self.__chull_prune_points(points)
        
            points = points[pruned_ind]
            
        except:
            pass
        
        points_zero = points[points[:,-1] > 0]
        
#         try:
#             if (self.pricer_options['convex_hull_filter'] == 'qhull') and (points_zero.shape[0] > x.shape[1]):
#                 points_zero = points_zero[ConvexHull(points_zero[:,:-1]).vertices]
#         except:
#             pass
            
        
        points_zero[:,-1] = 0.0
        
        points = np.vstack((points,points_zero))
        
        if self.pricer_options['convex_hull_filter'] == 'qhull':
            
#             with Timer('Convex hull', flush=True):
            cv_point_indices = ConvexHull(points).vertices
#                 print('result = {0}/{1}'.format(cv_point_indices[cv_point_indices < x.shape[0]].shape[0], points.shape[0]))
                
#             raise Exception('stopped')
        
#             print('x.shape[0] = ', x.shape[0])
#             print('cv_point_indices = ', cv_point_indices)
#             print('pruned_ind', pruned_ind)
            return pruned_ind[cv_point_indices[cv_point_indices < pruned_ind.shape[0]]]
        
        
        raise ValueError('unknown convex_hull_filter value \'{0}\''.format(self.pricer_options['convex_hull_filter']))
        

    def find_u(self, x, v, z):

        # flat surface
        if np.max(v)-np.min(v) < 1e-15:
            
            ind = np.argmin(np.max(np.abs(x-z), axis=1))
            Qopt = 0*v
            Qopt[ind] = 1
            
#             return (np.mean(v), Qopt)

            return (np.mean(v), np.nan)
            
        
        try:
            ind = self.__get_cvhull_indices(x, v)
#             print(np.hstack((x[ind], v[ind].reshape(-1,1))))
#             print('z = ', z)
            
        except Exception as ex:
            
            print('x = ', x)
            print('v = ', v)

            raise ex
        
        try:
            Qopt = get_max_coordinates(x[ind], v[ind], z, debug_mode=self.debug_mode, ignore_warnings=self.ignore_warnings)
#             print('Qopt = ', Qopt)

        except Exception as ex:

#             print('[(n, x, v)] = ')
#             for c in zip(range(x.shape[0]),x,v): print(c)
            
            print('convex hull [(n, x, v)] = ')
            for c in zip(ind, x[ind],v[ind]): print(c)
                
            print('z = ', z)

            raise ex


#         print('Qopt = ', Qopt)
#         print('v[ind] = ', v[ind])
        Vopt = Qopt @ v[ind]

        return (Vopt, Qopt)


    def find_rho(self, x, v, K_x, convdK_x):
        
        convdK_x = np.atleast_2d(convdK_x)
        K_x      = np.atleast_2d(K_x)

        supp_func = self.constraint_set.support_function(convdK_x - (1 if self.lattice.logscale else 0))
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
            
            Vopt, _ = self.find_u(x, v, K_x[i])
            
            res_u[i] = Vopt
        
        maxind = np.argmax(res_u - supp_func)
        
        return (res_u[maxind] - supp_func[maxind], convdK_x[maxind])
    
    
    def evaluate(self):
        
        with PTimer(header='Init stage of evaluate()', silent=True, profiler_data=self.profiler_data) as tm:
            self.__precalc()
        
        with Timer('Main stage of evaluate()', flush=False, silent=self.silent_timer_) as tm_total:

            pdata = self.profiler_data.data[tm_total.header]
            
            with Timer('Precalculation of the evaluation points for the value function', silent=self.silent_timer_) as tm:
                Vp, Vf = self.generate_evaluation_point_lists(self.p0_, self.dK_, self.N, profiler_data=pdata.data[tm.header])


            with PTimer('Evaluation of the value function at the terminal moment', silent=self.silent_timer_,
                      profiler_data=pdata) as tm:

                x = self.lattice.map2x(Vp[-1])
                    
                Vf[-1] = self.option.payoff(x)
                      

            with Timer('Evaluation of the value function at the intermediate moments', silent=self.silent_timer_) as tm:
                
                pdata2 = pdata.data[tm.header]
                
                for t in reversed(range(self.N)):
                    
                    if not self.silent_timer_: print('t = {0}'.format(t))
                    
                    res = np.empty(Vp[t].shape[0], dtype=Vf[t+1].dtype)
                    
                    for i, vp in enumerate(Vp[t]):
                        
                        if not self.silent_timer_:
                            if (np.random.uniform()<0.001): print('iter = {0}/{1}'.format(i, len(Vp[t])))

                        with PTimer(header='K = vp + self.dK_', silent=True, profiler_data=pdata2) as tm2:
                            
                            K = vp + self.dK_

                        with PTimer(header='tf = isin_points(Vp[t+1], K)', silent=True, profiler_data=pdata2) as tm2:
                            
                            tf = isin_points(Vp[t+1], K)

                        with PTimer(header='find_rho', silent=True, profiler_data=pdata2) as tm2:
        
                            res_v, _ = self.find_rho(self.lattice.map2x(Vp[t+1][tf]), Vf[t+1][tf], self.lattice.map2x(K), self.lattice.map2x(self.dK_))
                            res[i] = res_v
                
#                             print('vp = ', self.lattice.map2x(vp))
#                             print('res_v = ', res_v)
#                             print('Vp[t+1], Vf[t+1] = ')
#                             for c1, c2 in zip(self.lattice.map2x(Vp[t+1][tf]), Vf[t+1][tf]):
#                                 print(c1, c2)

                    Vf[t] = res                
                
        gc.collect()

        return Vf[0][0]