# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

from sklearn.utils import check_random_state
import numpy as np

from .util import nearest, farthest, distance
from .quadprog import quad_prog


__all__ = [
          'triangle_algorithm',
          'AVTA'
          ]


def __in_cv_check(v, v_new, p1, p, eps, debug_mode):

    n = 3

    X = np.vstack((v,v_new,p1)).T
    f = -2*np.dot(X.T, p)
    c = np.dot(p,p)
    H = 2*np.dot(X.T, X)
    A = np.eye(n)
    lb = np.zeros(n, dtype=np.float64)
    ub = np.ones(n, dtype=np.float64)
    B = np.ones(n, dtype=np.float64)
    b = float(1)
    x0 = np.ones(n, dtype=np.float64) / float(n)
    
    opts = {'disp': debug_mode, 'gtol': 0.01*eps}

    res = quad_prog(x0, H, f, c, A, lb, ub, B, b, options=opts)
    
    return (res.status < 3) and (np.sqrt(res.fun) < eps)


def triangle_algorithm(S, p, eps=1e-8, random_state = None, max_iter=10000, debug_mode=False, cv_check=False, cv_check_interval=100):
    
    if random_state is None:
        random_state = check_random_state(random_state)
    
    assert eps > 0, "eps is <= 0"
    assert eps < 1, "eps is >= 1"
    
    S = list(S)
    
    
    # Step 0
    
    ind, _ = nearest(p,S)
    
    p1 = S[ind]
    v  = S[ind]
    
    alpha = np.zeros(len(S), dtype=np.float32)
    alpha[ind] = 1.0
    
    iter = 0
    cv_check_iter=0
    
    max_iter_flag = (max_iter is not None)
    max_iter = (max_iter if max_iter is not None else 1)
    
    if debug_mode:
        print('** START triangle algorithm **')
        print('p = ', p)
        print('S = ', S)
    
    while iter < max_iter:
    
        if debug_mode:
            print('-- iter {0} --'.format(iter))
            print('\t p1 = ', p1)
            print('\t v = ', v)
        
#         Step 1

        if distance(p, p1) <= (eps * distance(p, v)):
            return (True, p1) # eps-approximate solution found
        
        p_pivot = None
        
        random_state.shuffle(S)
        
        for ind_pivot, v_new in enumerate(S):
            
            if distance(p1, v_new) >= distance(p, v_new):
                p_pivot = v_new
                break
        
        if debug_mode: print('\t p_pivot = ', p_pivot)
            
        if p_pivot is None: # no pivot exists
            
            if debug_mode: print('** END triangle algorithm **')
                
            return (False, p1)
        
        if debug_mode:
            print('\t v_new = ', v_new)
          
        if cv_check:
            
            if cv_check_iter < cv_check_interval:
                cv_check_iter += 1
                
            else:
                if __in_cv_check(v, v_new, p1, p, eps, debug_mode):

                    if debug_mode:
                        print('in_cv check successfull')
                        print('** END triangle algorithm **')

                    return (True, np.nan)
                
                cv_check_iter = 0
            
        v = v_new
        
        # Step 2
        
        d = distance(p1, v)
        
        if d == 0:
            beta = 0
        else:
#             beta = np.clip(np.dot((v - p) / d, (v - p1) / d) , 0, 1)
            beta = np.dot((v - p) / d, (v - p1) / d)
            
        for i, al in enumerate(alpha):
            
            if (i == ind_pivot) and (al < 1.0):
                beta = np.clip(beta, 0, float(1) / (float(1) - al))
                
            if (i != ind_pivot) and (al > 0.0):
                beta = np.clip(beta, 0, float(1) / al)                
            
        p1 = beta * p1 + (1 - beta) * v
        
        if debug_mode: print('\t beta = ', beta)
        
        for i in range(len(S)):
            if i == ind_pivot:
                alpha[i] = beta * alpha[i] + 1.0 - beta
            else:
                alpha[i] *= beta
        
        if max_iter_flag:
            iter += 1
    
    raise RuntimeWarning('triangle algorithm: method has not converged, maximum number of iterations has been exceeded')
    
    if debug_mode:
        print('** END triangle algorithm **')
    
    
def AVTA(S, gamma=1e-8, random_state=None, max_iter=10000, debug_mode=False, TA_cv_check=False, TA_cv_check_interval=100):
    
    if random_state is None:
        random_state = check_random_state(random_state)
    
    assert gamma > 0, "gamma is <= 0"
    assert gamma < 1, "gamma is >= 1"
    
    S = list(S)
    S_ind = list(range(len(S)))
    S_hat_ind = []
#     S_hat = []
    
    if debug_mode:
        print('S = ')
        for i, s in enumerate(S):
            print('{0}\t{1}'.format(i,s))
        print('')
    
    # Step 0
    
    v = S[random_state.choice(S_ind)]
    
    ind, _ = farthest(v, S)
    
    S_hat_ind.append(ind)    
#     S_hat.append(S[ind])
    
    max_iter_flag = max_iter is not None
    max_iter_ = max_iter if max_iter is not None else 1
    
    iter1 = 0
    goto_step1_flag = True
    
    while iter1 < max_iter_:
        
        if max_iter_flag:
            iter1 += 1
            
        if debug_mode:
            print('-- Outer loop, iter {0} --'.format(iter1))
    
        # Step 1
        
        if goto_step1_flag:
            
            if debug_mode:
                print('\t Step 1')
                print('\t S_ind = ', S_ind)
                print('\t S_hat_ind = ', S_hat_ind)

            I = [i for i in S_ind if i not in S_hat_ind]
            
            if len(I) == 0:
                return [S[i] for i in S_hat_ind]
            
            ind = random_state.choice(I)
            v = S[ind]
            
        # Step 2

        if debug_mode: print('\t Step 2')

        in_hull, p1 = triangle_algorithm([S[i] for i in S_hat_ind], v, 0.5 * gamma, max_iter=max_iter,
                                         random_state=random_state, debug_mode=debug_mode, cv_check=TA_cv_check, cv_check_interval=TA_cv_check_interval)

        if debug_mode:
            print('\t in_hull = ', 'True' if in_hull else 'False')
            print('\t p1 = ', p1)

        # Step 3

        if debug_mode: print('\t Step 3')

        if in_hull:

            if debug_mode: print('\t from S_ind pop ind = ', ind)

            S_ind.remove(ind)
            
            if debug_mode: print('\t changed S_ind = ', S_ind)

            if len(S_ind) == 0:

                if debug_mode: print('\t return S_hat_ind = ', S_hat_ind)

                return [S[i] for i in S_hat_ind]

            else:

                if debug_mode: print('\t go to Step 1')

                goto_step1_flag = True
                continue

        else:
            
            if debug_mode: print('\t go to Step 4')

        # Step 4
        
        if debug_mode: print('\t Step 4')

        c1 = v - p1

        S1_ind = []
        d1 = -np.Inf

        for ind in [i for i in S_ind if i not in S_hat_ind]:

            tmp = np.dot(c1, S[ind])

            if tmp > d1:
                d1 = tmp
#                     S1 = [S[ind]]
                S1_ind = [ind]

            elif tmp == d1:
#                     S1.append(S[ind])
                S1_ind.append(ind)
    
        if debug_mode: print('\t S1_ind = ', S1_ind)

        v1_ind, _ = farthest(S[random_state.choice(S1_ind)], [S[i] for i in S1_ind])
        
        if debug_mode: print('\t v1_ind = ', v1_ind)
        if debug_mode: print('\t S1_ind[v1_ind] = ', S1_ind[v1_ind])
        
        v1 = S[S1_ind[v1_ind]]

        S_hat_ind.append(S1_ind[v1_ind])
        
        if debug_mode: print('\t changed S_hat_ind = ', S_hat_ind)

        # Step 5
        
        if debug_mode: print('\t Step 5')

        if np.all(v == v1):
            if debug_mode: print('\t go to Step 1')
            goto_step1_flag = True
        else:
            if debug_mode: print('\t go to Step 2')
            goto_step1_flag = False
        
    
    raise RuntimeWarning('method has not converged, maximum number of iterations has been exceeded')
    
    