# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import numpy as np
from scipy.optimize import minimize, LinearConstraint


__all__ = [
          'quad_prog'
          ]


def __transform_matrix(A):
    
    A = np.asarray(A)
    
    if len(A.shape) == 0:
        A = A.reshape(-1,1)
    elif len(A.shape) == 1:
        A = A.reshape(1,-1)
        
    return A


def quad_prog(x0, H, f, c, A, lb, ub, B, b, **kwargs):
    """
    Solves the constrained quadratic programming problem using Scipy's 'trust-constr' method.
    The problem for x is:
    0.5 * <x, Hx> + <b, x> + c -> min,
    lb <= Ax <= ub,
    Bx = b
    """
    
    H = __transform_matrix(H)
    A = __transform_matrix(A)
    B = __transform_matrix(B)
    
    fun = lambda x: 0.5 * np.dot(x, np.dot(H,x)) + np.dot(f, x) + c
    
    jac = lambda x: 0.5 * np.dot(H + H.T, x) + f
    
    hess = lambda x: 0.5 * (H + H.T)
    
    constr_ineq = LinearConstraint(A, lb, ub)
    
    constr_eq = LinearConstraint(B, b-1e-12, b+1e-12)

#     iter = 0
#     def callback_F(x, res):
#         global iter
#         iter += 1
#         print('x = ', x)
#         print('jac = ', jac(x))
#         print('hess = ', hess(x))
    
    return minimize(fun, x0, method='trust-constr', jac=jac, hess=hess, constraints=[constr_ineq, constr_eq], **kwargs)
    