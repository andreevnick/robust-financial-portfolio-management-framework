# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

from datetime import datetime
import time
import collections
import numpy as np

__all__ = ['tic',
           'toc',
           'coalesce',
           'keydefaultdict',
           'cartesian_product',
           'triple_product',
           'cantor_pairing_function',
           'pairing_function',
           'get_support_set',
           'generate_evaluation_point_lists'
          ]

from guaranteed.pricing import PIDynamics
from guaranteed.util import minksum_points, minkprod_points, PTimer

__timestamp = None;


def tic():

    global __timestamp
    
    __timestamp = datetime.now()
    
    
def toc():
    
    global __timestamp
    
    sec = (datetime.now() - __timestamp).total_seconds()
    mnt = sec // 60
    sec = sec - 60 * mnt
    
    if mnt > 0:
        res = 'Время расчетов: {0} мин {1:.4f} сек'.format(mnt, sec)
    else:
        res = 'Время расчетов: {0:.4f} сек'.format(sec)
        
#     print(res)
    
    return (res, sec)
    

def coalesce(*arg):
    return next((a for a in arg if a is not None), None);


class keydefaultdict(collections.defaultdict):
    
    def __missing__(self, key):
        
        if self.default_factory is None:
            raise KeyError( key )
            
        else:
            ret = self[key] = self.default_factory(key)
            return ret
        
        
def cartesian_product(*arrays):
    '''
    https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    '''
    
    la = len(arrays)
    
    dtype = np.result_type(*arrays)
    
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
        
    return arr.reshape(-1, la)


def triple_product(a, b, c):
    '''
    a,b,c - [n x 3] numpy arrays
    Return: [n x 1] numpy array of triple product values
    '''
    
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    
    return a[:,0] * (b[:,1] * c[:,2] - b[:,2] * c[:,1]) + a[:,1] * (b[:,2] * c[:,0] - b[:,0] * c[:,2]) + a[:,2] * (b[:,0] * c[:,1] - b[:,1] * c[:,0])

    
def __Z_to_N(x):
    
    tf = x<=0
    
    res = np.nan * x
    
    res[tf] = 2*(-x[tf]) + 1
    res[~tf] = 2 * x[~tf]
    
    return res


def __N_to_Z(x):
    
    tf = (x // 2).astype(x.dtype) == 0
    
    res = np.nan * x
    
    res[tf] = (x[tf] // 2).astype(x.dtype)
    res[~tf] = -((x[~tf]-1) // 2).astype(x.dtype)
    
    return res
    

def cantor_pairing_function(x1, x2):
    '''
    https://en.wikipedia.org/wiki/Cantor_pairing_function
    '''
    
    x1 = __Z_to_N(x1)
    x2 = __Z_to_N(x2)
    
    return (((x1 + x2) * (x1 + x2 + 1)) // 2 + x2).astype(x1.dtype)


# def inverse_cantor_pairing_function(x):
#     '''
#     https://en.wikipedia.org/wiki/Cantor_pairing_function
#     '''
        
#     # !! UNTESTED !!
    
#     w = np.floor(0.5 * (np.sqrt(8 * x + 1) - 1))
    
#     x2 = x - (0.5 * (w*w + w)).astype(x.dtype)
#     x1 = w - x2
    
#     return (__N_to_Z(x1), __N_to_Z(x2))


def pairing_function(x):
    
    x_ = np.atleast_2d(x)
    
    if x_.shape[1] <= 1:
        return x
    
    if x_.shape[0] == 0:
        return np.zeros(1, dtype=x.dtype)
    
    if x_.shape[1] > 2:
        return cantor_pairing_function(pairing_function(x_[:,:-1]), x_[:,-1])
    
    else:
        return cantor_pairing_function(x_[:,0], x_[:,1])


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
        raise ValueError('This algorithm only works for price-independent dynamics!')


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