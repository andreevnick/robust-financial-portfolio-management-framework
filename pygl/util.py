import numpy as np


__all__ = ['coalesce',
           'cartesian_product',
           'pairing_function'
          ]


def coalesce(*arg):
    return next((a for a in arg if a is not None), None);


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


def __cantor_pairing_function(x1, x2):
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
        return __cantor_pairing_function(pairing_function(x_[:,:-1]), x_[:,-1])
    
    else:
        return __cantor_pairing_function(x_[:,0], x_[:,1])
    