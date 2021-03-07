# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import numpy as np
from ..util import coalesce, cartesian_product


__all__ = [
            'ISetHandler',
            'SetHandler',
            'RectangularHandler',
            'EllipseHandler',
            'RealSpaceHandler',
            'NonNegativeSpaceHandler'
          ]


class ISetHandler:
    """ Interface which contains the set-based operations for the implementing set.
    
    """
    
    def project(self, grid):
        """ Projects the set to the grid.

        Parameters
        ----------
        grid : class:`Grid`
            Point grid.

        Returns
        -------
        numpy.ndarray
            Array of points on the grid which belong to the set.

        """
        
        raise Exception("The method must be defined in a subclass")
        
        
    def support_function(self, x):
        """ Support function of the set at x.

        Parameters
        ----------
        x : numpy.ndarray, size = (m,n)
            Array of points from R^n.

        Returns
        -------
        numpy.ndarray
            Array of support function values, might be np.Inf.

        """
        
        raise Exception("The method must be defined in a subclass")
        
    
    def iscompact(self):
        """ Checks if the set is compact.

        Returns
        -------
        boolean
            True if the set is compact.

        """
        
        raise Exception("The method must be defined in a subclass")
        
        
    def multiply(self, x):
        """ Multiplies the set by a point x from R^n.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        ISetHandler
            Set object which corresponds to the set {x*a, a \in A} where A is the current set.

        """
        
        raise Exception("The method must be defined in a subclass")
        
        
        
class SetHandler(ISetHandler):
    """ SetHandler factory, will be deprecated, not recommended for usage.
    
    """
    
    def __init__(self, shape, verify_compactness=False, **kwargs):
                
        if shape.lower() == 'rect':
            self.handler = RectangularHandler(kwargs.pop('bounds'))
            
        elif shape.lower() == 'ellipse':
            self.handler = EllipseHandler(kwargs.pop('mu'), kwargs.pop('sigma'), kwargs.get('conf_level', None))
            
        elif shape.lower() == 'unbounded':
            self.handler = RealSpaceHandler()
            
        elif shape.lower() == 'nonnegative':
            self.handler = NonNegativeSpaceHandler()
            
        else:
            raise ValueError('Unsupported constraint type \'{0}\''.format(shape))
            
        if verify_compactness:
            assert self.handler.iscompact(), 'The set is not compact'
            


class RectangularHandler(ISetHandler):
    
    def __init__(self, bounds, log=False, dtype=None):
        
        self.bounds = np.asarray(bounds, dtype=coalesce(dtype, np.float64))
        
        
    def project(self, grid):

        bnd = np.vstack((grid.get_point(self.bounds.T[0]), grid.get_point(self.bounds.T[1]))).T

        return cartesian_product(*[np.arange(lb, ub+1) for lb, ub in bnd])
    
    
    def support_function(self, x):
        
        x = np.atleast_2d(x)

        return np.sum(np.maximum(x * np.tile(self.bounds.T[0], (x.shape[0],1)),
                                 x * np.tile(self.bounds.T[1], (x.shape[0],1))),
                      axis=1)
    
    
    def iscompact(self):
        
        return not np.any(np.isinf(self.bounds))
    
    
    def multiply(self, x):
        
        assert np.all(x >= 0), 'x must be >= 0'
        
        return RectangularHandler(self.bounds * np.asarray(x).reshape(-1,1), dtype=self.bounds.dtype)
    


class EllipseHandler(ISetHandler):
    
    def __init__(self, mu, sigma, conf_level=None, dtype=None):
        
        self.mu = np.atleast_2d(mu).astype(coalesce(dtype, np.float64))
        
        self.sigma = np.atleast_2d(sigma).astype(coalesce(dtype, np.float64))
        
        if conf_level is not None:
            # https://stats.stackexchange.com/questions/64680/how-to-determine-quantiles-isolines-of-a-multivariate-normal-distribution
            self.sigma *= -2 * np.log(1-conf_level)
        
        eigv, U = np.linalg.eig(self.sigma)
        self.L = np.sqrt(np.diag(eigv))
        self.U = U
        
        self.sigma_inv = np.linalg.inv(self.sigma)
        
                
        
    def __r2(self, x):

        return np.sum((x-self.mu).dot(self.sigma_inv) * (x-self.mu), axis=1)
    
        
        
    def project(self, grid):
        
        R = np.max(np.abs(np.diagonal(self.L))) + np.max(grid.delta)
        
        S = RectangularHandler(self.mu.T + np.array([-R, R], dtype=self.L.dtype)).project(grid)

        return S[self.__r2(grid.map2x(S)) <= 1]
    
    
    def support_function(self, x, x_center=None):
        
        x = np.atleast_2d(x)
            
        return (x @ self.mu) + np.linalg.norm(self.L @ self.U.T @ x.T, axis=0, ord=2)
    
    
    def iscompact(self):
        
        return (not np.any(np.isinf(self.sigma))) and (not np.any(np.isinf(self.mu)))
    
    
    def multiply(self, x):
        
        assert np.all(x >= 0), 'x must be >= 0'
        
        D = np.diag(x)
        
        return EllipseHandler(self.mu @ D,  D @ self.sigma @ D.T, dtype=np.result_type(self.mu, self.sigma))
    
    
    
class RealSpaceHandler(ISetHandler):
    
    def __init__(self):
        
        pass
    
    
    def project(self, grid):
        
        raise Exception('Cannot map the unbounded set')
        
        
    def support_function(self, x):
        
        x = np.atleast_2d(x)
            
        return np.where(np.all(x == 0, axis=1), 0, np.Inf)
    
    
    def iscompact(self):
        
        return False
    
    
    def multiply(self, x):
        
        return RealSpaceHandler()
    
    
    
class NonNegativeSpaceHandler(ISetHandler):
    
    def __init__(self):
        
        pass
    
    
    def project(self, grid):
        
        raise Exception('Cannot map the unbounded set')
        
        
    def support_function(self, x):
        
        x = np.atleast_2d(x)
        
        return np.where(np.all(x <= 0, axis=1), 0, np.Inf)
    
    
    def iscompact(self):
        
        return False
    
    
    def multiply(self, x):
        
        return NonNegativeSpaceHandler()
    
    