import numpy as np

from .scene_object import SceneObject


__all__ = ['ScatterObject'
          ]



class ScatterObject(SceneObject):
    
    def __init__(self, points=[], **kwargs):
        
        SceneObject.__init__(self, **kwargs)
        
        self.points = np.asarray(points)
        
