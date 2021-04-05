import numpy as np

from .util import coalesce


__all__ = ['Grid']


class Grid:
    
    def __init__(self, delta, logscale=False, center=None, dtype=None, dtype_p=None):
        
        self.dtype = coalesce(dtype, np.float64)
        self.dtype_p = coalesce(dtype_p, np.int64)
        self.delta = np.asarray(delta, dtype=self.dtype)        
        self.logscale = logscale        
        self.center = np.asarray(coalesce(center, self.delta*0), dtype=self.dtype)
        
    def x_trans(self, x):
        
        return x if self.logscale == False else np.log(x)
    
    
    def x_trans_inv(self, x):
        
        return x if self.logscale == False else np.exp(x)

        
    def get_projection(self, obj, **kwargs):
        
        # obj is an array of coordinates
            
        return self.get_point(obj)
        
        
    def get_point(self, x):
        
        x = self.x_trans(np.asarray(x, dtype=self.dtype))
        
        return np.rint((x - self.center)/self.delta).astype(self.dtype_p)
    
    
    def map2x(self, point):
                    
        return self.x_trans_inv(point * self.delta + self.center)
        
        