import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches

from scipy.spatial import ConvexHull

import copy

from .util import *



__all__ = ['ASceneObject',
           'GridObject',
           'ScatterObject',
           'LineObject',
           'LevelLineObject',
           'SurfaceObject',
           'APatchObject',
           'EllipsePatchObject',
           'RectanglePatchObject',
           'PathPatchObject'
          ]


class ASceneObject:
    
    def __init__(self, plot_last=False, visible=True, **kwargs):
        
        self.plot_last = plot_last
        self.visible = visible
        self.opts = kwargs.copy()
        
        
    def draw_in_axes(self, ax):
        
        raise Exception('Must be redefined in a subclass')
        
        
        
class GridObject(ASceneObject):
    
    def __init__(self, delta, logscale=False, center=None, dtype=None, dtype_p=None, **kwargs):
        
        ASceneObject.__init__(self, **kwargs)
        
        self.dtype = coalesce(dtype, np.float64)
        self.dtype_p = coalesce(dtype_p, np.int64)
        self.delta = np.asarray(delta, dtype=self.dtype)        
        self.logscale = logscale        
        self.center = np.asarray(coalesce(center, self.delta*0), dtype=self.dtype)
        
    def x_trans(self, x):
        
        return x if self.logscale == False else np.log(x)
    
    
    def x_trans_inv(self, x):
        
        return x if self.logscale == False else np.exp(x)
        
        
    def get_point(self, x):
        
        x = self.x_trans(np.asarray(x, dtype=self.dtype))
        
        return np.rint((x - self.center)/self.delta).astype(self.dtype_p)
    
    
    def map2x(self, point):
                    
        return self.x_trans_inv(point * self.delta + self.center)
    
    
    def draw_in_axes(self, ax):
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        if is3D(ax):
            
            zlim = ax.get_zlim()
            
            p0 = self.get_point([xlim[0], ylim[0], zlim[0]])
            p1 = self.get_point([xlim[1], ylim[1], zlim[1]])
        
            points = cartesian_product(np.arange(p0[0], p1[0]+1), np.arange(p0[1], p1[1]+1), np.arange(p0[2], p1[2]+1))
            
        else:
            
            p0 = self.get_point([xlim[0], ylim[0]])
            p1 = self.get_point([xlim[1], ylim[1]])
        
            points = cartesian_product(np.arange(p0[0], p1[0]+1), np.arange(p0[1], p1[1]+1))
            
        ScatterObject(self.map2x(points), **self.opts).draw_in_axes(ax)
    

    
class ScatterObject(ASceneObject):
    
    def __init__(self, points=[], **kwargs):
        
        ASceneObject.__init__(self, **kwargs)
        
        self.points = np.asarray(points)
        
        
    def draw_in_axes(self, ax):
        
        if self.points.shape[1] == 2:
        
            ax.scatter(self.points[:,0], self.points[:,1], **self.opts)
            
        else:
            
            ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], **self.opts)
        

        
class LineObject(ASceneObject):
    
    def __init__(self, points=[], **kwargs):
        
        ASceneObject.__init__(self, **kwargs)
        
        self.points = np.asarray(points)
        
        
    def draw_in_axes(self, ax):
        
        if self.points.shape[1] == 2:
        
            ax.plot(self.points[:,0], self.points[:,1], **self.opts)
            
        else:
            
            ax.plot(self.points[:,0], self.points[:,1], self.points[:,2], **self.opts)
            

            
class LevelLineObject(ASceneObject):
    
    def __init__(self, x=None, y=None, z=None, **kwargs):
        
        ASceneObject.__init__(self, plot_last=True, **kwargs)
        
        self.x = x
        self.y = y
        self.z = z
        
        
    def draw_in_axes(self, ax):
        
        if is3D(ax):
            
            if (self.x is not None) and (self.y is not None) and (self.z is None):
                lim = ax.get_zlim()
                ax.plot([self.x, self.x], [self.y, self.y], lim, **self.opts)
                
            elif (self.x is not None) and (self.y is None) and (self.z is not None):
                lim = ax.get_ylim()
                ax.plot([self.x, self.x], lim, [self.z, self.z], **self.opts)
                
            elif (self.x is None) and (self.y is not None) and (self.z is not None):
                lim = ax.get_xlim()
                ax.plot(lim, [self.y, self.y], [self.z, self.z], **self.opts)
                
            else:
                raise Exception('Cannot recognize the type of level object')
                
        else:
            
            if (self.x is not None) and (self.y is None):
                lim = ax.get_ylim()
                ax.plot([self.x, self.x], lim, **self.opts)
                
            elif (self.x is None) and (self.y is not None):
                lim = ax.get_xlim()
                ax.plot(lim, [self.y, self.y], **self.opts)
                
            else:
                raise Exception('Cannot recognize the type of level object')
                

                
class SurfaceObject(ASceneObject):
    
    def __init__(self, X, Y, Z, anchor_Xgrid=None, anchor_Ygrid=None, wireframe=False, **kwargs):
        
        ASceneObject.__init__(self, **kwargs)
        
        self.wireframe = wireframe
        
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        Z = np.atleast_1d(Z)
        
        if len(Z.shape) == 1:
            
            Xgrid, Ygrid, Zgrid = XYZ_to_grid_interp(X, Y, Z, Xgrid=anchor_Xgrid, Ygrid=anchor_Ygrid)
            
            self.Xgrid = Xgrid
            self.Ygrid = Ygrid
            self.Zgrid = Zgrid
            
        else:
            
            self.Xgrid = X
            self.Ygrid = Y
            self.Zgrid = Z
        
        
    def draw_in_axes(self, ax):
        
        if not is3D(ax):
        
            raise Exception('SurfaceObject must be placed in 3D axes.')
            
        else:
            
            if self.wireframe:
                ax.plot_wireframe(self.Xgrid, self.Ygrid, self.Zgrid, **self.opts)
            else:                
                ax.plot_surface(self.Xgrid, self.Ygrid, self.Zgrid, **self.opts)
            
            
            
class APatchObject(ASceneObject):
    
    def __init__(self, patch=None, **kwargs):
        
        ASceneObject.__init__(self, **kwargs)
        
        self.patch = copy.deepcopy(patch)
        
        
    def draw_in_axes(self, ax):
        
        ax.add_patch(self.patch)
        
        
        
class EllipsePatchObject(APatchObject):
    
    def __init__(self, center, semi_axes, angle=0.0, **kwargs):
        
        assert np.asarray(center).shape[0] == 2, 'EllipsePatchObject is not supported for 3D axes'
        
        APatchObject.__init__(self, **kwargs)
        
        self.center = np.asarray(center).flatten()
        
        if np.isscalar(semi_axes):
            self.semi_axes = semi_axes + np.zeros(len(center), dtype=center.dtype)
        else:
            self.semi_axes = np.asarray(semi_axes).flatten()
            
        assert np.all(self.semi_axes) >= 0, 'semi_axes must be >= 0'
            
        self.angle = angle
        
        self.patch = mpatches.Ellipse(self.center, 2*self.semi_axes[0], 2*self.semi_axes[1], self.angle, **kwargs)
        
        
        
class RectanglePatchObject(APatchObject):
    
    
    def __init__(self, corner, width, height, angle=0.0, **kwargs):
        
        assert np.asarray(corner).flatten().shape[0] == 2, 'RectanglePatchObject is not supported for 3D axes'
        
        APatchObject.__init__(self, **kwargs)
        
        self.corner = np.asarray(corner).flatten()
        self.width = width
        self.height = height
            
        assert self.width  >= 0, 'width must be >= 0'
        assert self.height >= 0, 'height must be >= 0'
            
        self.angle = angle
        
        self.patch = mpatches.Rectangle(self.corner, self.width, self.height, self.angle, **kwargs)



class PathPatchObject(APatchObject):
    
    def __init__(self, points = [], is_convex=False, **kwargs):
        
        APatchObject.__init__(self, **kwargs)
        
        self.points = np.asarray(points)
        
        if self.points.shape[0] > 0:
            assert self.points.shape[1] == 2, 'RectanglePatchObject is not supported for 3D axes'
        
        self.is_convex = is_convex
        
        self.patch = self.__get_patch() if not self.is_convex else self.__get_convex_patch()
    
    
    @staticmethod
    def __get_path(points):
        
        Path = mpath.Path
        
        path_data = []
        
        for p_num, p in enumerate(points):
            
            if p_num == 0:
                path_data.append((p, Path.MOVETO))
                
            else:
                path_data.append((p, Path.LINETO))
                
        path_data.append((points[-1], Path.CLOSEPOLY))
        
        return mpath.Path(*zip(*path_data))
        
        
    def __get_patch(self):
        
        return mpatches.PathPatch(PathPatchObject.__get_path(self.points), **self.opts)
        
        
    def __get_convex_patch(self):
        
        c = self.points.mean(axis=0)
        
        angles = np.arctan2(self.points[:,1]-c[1], self.points[:,0] - c[0])
        
        p_array = self.points[np.flip(np.argsort(angles))]
        
        return mpatches.PathPatch(PathPatchObject.__get_path(p_array), **self.opts)
            