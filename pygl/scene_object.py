__all__ = ['SceneObject'
          ]


class SceneObject:
    
    def __init__(self, visible=True, **kwargs):
        
        self.visible=visible
        
        self.opts = kwargs
        
        
