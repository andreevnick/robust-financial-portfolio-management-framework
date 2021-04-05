import numpy as np

import matplotlib.path as mpath
import matplotlib.patches as mpatches

from scipy.spatial import ConvexHull

import copy


from .scene_object import SceneObject


__all__ = ['PatchObject',
           'EllipsePatchObject',
           'RectanglePatchObject',
           'PathPatchObject'
          ]


class PatchObject(SceneObject):
    
    def __init__(self, patch=None, **kwargs):
        
        SceneObject.__init__(self, **kwargs)
        
        self.patch = copy.deepcopy(patch)
        
        
        
class EllipsePatchObject(PatchObject):
    
    
    def __init__(self, center, semi_axes, angle=0.0, **kwargs):
        
        SceneObject.__init__(self, **kwargs)
        
        self.center = np.asarray(center).flatten()
        
        if np.isscalar(semi_axes):
            self.semi_axes = semi_axes + np.zeros(len(center), dtype=center.dtype)
        else:
            self.semi_axes = np.asarray(semi_axes).flatten()
            
        assert np.all(self.semi_axes) >= 0, 'semi_axes must be >= 0'
            
        self.angle = angle
        
        self.patch = mpatches.Ellipse(self.center, 2*self.semi_axes[0], 2*self.semi_axes[1], self.angle, **kwargs)
        
        
        
class RectanglePatchObject(PatchObject):
    
    
    def __init__(self, corner, width, height, angle=0.0, **kwargs):
        
        SceneObject.__init__(self, **kwargs)
        
        self.corner = np.asarray(corner).flatten()
        self.width = width
        self.height = height
            
        assert self.width  >= 0, 'width must be >= 0'
        assert self.height >= 0, 'height must be >= 0'
            
        self.angle = angle
        
        self.patch = mpatches.Rectangle(self.corner, self.width, self.height, self.angle, **kwargs)



class PathPatchObject(PatchObject):
    
    def __init__(self, points = [], is_convex=False, **kwargs):
        
        SceneObject.__init__(self, **kwargs)
        
        self.points = np.asarray(points)
        
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
    
