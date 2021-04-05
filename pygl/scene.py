import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from os import path

from .grid import Grid
from .scatter import ScatterObject
from .patch import PatchObject
from .line import LineObject
from .scene_object import SceneObject

from .util import cartesian_product
    
    
__all__ = [
            'Scene',
            'SceneObjectFactory'
          ]

    
class Scene:
    
    def __init__(self, **kwargs):
        
        conf = self.__read_conf(kwargs.get('template', path.join(path.dirname(__file__), '.\scene_templates\default.json')))
        override_conf = kwargs.get('override_conf', {})
            
        self.conf = conf['scene2D']
        
        for k, v in override_conf.items():
            self.conf[k] = v
        
        self.xlabel = kwargs.get('xlabel', '')
        self.ylabel = kwargs.get('ylabel', '')
        
        self.title = kwargs.get('title', '')
        
        self.xlim = kwargs.get('xlim', '')
        self.ylim = kwargs.get('ylim', '')
        
        self.legend_loc = kwargs.get('legend_loc', 'best')
        
        self.init()
        
        
    def __read_conf(self, filename):
        
        if filename.split('.')[-1] != 'json':
            filename += '.json'
        
        try:
            with open(filename, mode='r') as f:
                conf = json.load(f)
                
        except Exception as ex:
            
            print('Cannot access \'{file}\''.format(file=filename))
            raise ex
            
        return conf
    
    
    def init(self):
        
        matplotlib.rc('font', size=self.conf['font'])
        
        self.scene_objects = []
        
        self.fig = plt.figure(figsize=self.conf['figsize'])
        
        self.ax = self.fig.gca()
        
        self.ax.grid(b=self.conf['grid'], which='major')
        self.ax.grid(b=self.conf['minor_grid'], which='minor')
        
        self.ax.set_xlim(self.xlim[0], self.xlim[1])
        self.ax.set_ylim(self.ylim[0], self.ylim[1])
        
        self.ax.set_title(self.title)
        
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        
        if 'background' in self.conf.keys():
            self.ax.set_facecolor(self.conf['background'])
        
        
    def append(self, obj, **kwargs):
        
        self.scene_objects.append(SceneObjectFactory.build(obj, **kwargs))
        
        
    def draw_scatter_object(self, obj):
        
        self.ax.scatter(obj.points[:,0], obj.points[:,1], **obj.opts)
        
        
    def draw_patch_object(self, obj):
        
        self.ax.add_patch(obj.patch)
        
        
    def draw_line_object(self, obj):
        
        self.ax.plot(obj.points[:,0], obj.points[:,1], **obj.opts)
        
        
    def draw_scene_object(self, obj):
                    
        if isinstance(obj, ScatterObject):
            
            self.draw_scatter_object(obj)
            
        elif isinstance(obj, PatchObject):
            
            self.draw_patch_object(obj)
            
        elif isinstance(obj, LineObject):
            
            self.draw_line_object(obj)
            
        else:
            
            print('Cannot draw a \'{0}\' object'.format(type(obj)))
    
    
    def show(self):
        
        for obj in self.scene_objects:
            
            if not obj.visible:
                obj.opts.pop('label', None)
                
            self.draw_scene_object(obj)
            
        if np.any([o.opts.get('label', None) is not None for o in self.scene_objects]):
            self.ax.legend(loc=self.legend_loc)
        
        plt.show()
        
            
        
        
class SceneObjectFactory:
    
    @staticmethod
    def build(obj, **kwargs):
        
        if isinstance(obj, Grid):
            
            return SceneObjectFactory.__construct_from_grid(obj, xlim=kwargs.pop('xlim'), ylim=kwargs.pop('ylim'), **kwargs)
        
        elif isinstance(obj, SceneObject):
            
            return SceneObjectFactory.__construct_from_sceneobj(obj, **kwargs)
            
        elif isinstance(obj, np.ndarray):
            
            return SceneObjectFactory.__construct_from_points(obj, **kwargs)
        
        else:
            
            raise TypeError('Cannot build from a \'{0}\' object'.format(type(obj)))
            
            
    @staticmethod
    def __construct_from_grid(grid, xlim, ylim, **kwargs):
        
        p0 = grid.get_point([xlim[0], ylim[0]])
        p1 = grid.get_point([xlim[1], ylim[1]])
        
        points = cartesian_product(np.arange(p0[0], p1[0]+1), np.arange(p0[1], p1[1]+1))
        
        return ScatterObject(points=grid.map2x(points), **kwargs)
    
    
    @staticmethod
    def __construct_from_points(points, **kwargs):
        
        return ScatterObject(points=points, **kwargs)
    
    
    @staticmethod
    def __construct_from_sceneobj(obj, **kwargs):
        
        return obj  

        
        