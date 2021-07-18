import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from os import path
import copy as cp

from .util import *
    
    
__all__ = [
            'Scene',
            'SceneGrid',
            'IsometricSceneGrid'
          ]

    
class Scene:
    
    def __init__(self, axes3D=False, ax=None, scene_objects=None, **kwargs):
        
        self.conf = self.__read_conf(kwargs.get('template', path.join(path.dirname(__file__), './scene_templates/default.json')))
#         self.conf = self.__read_conf(kwargs.pop('template', 'default.json'))
    
        for k, v in kwargs.items():
            self.conf[k] = v
            
        self.axes3D = axes3D
        
        self.ax = ax
        self.fig = None
        
        self.scene_objects = coalesce(scene_objects, [])
        
        
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
    
    
    def clone(self, deep=False):
        
        if not deep:
            return cp.copy(self)
        else:
            return Scene(axes3D=self.axes3D, ax=self.ax, scene_objects=self.scene_objects, **self.conf)
        
    
    def init(self):
        
        if 'font' in self.conf.keys():
            matplotlib.rc('font', size=self.conf['font'])
        
        if self.ax is None:
            
            self.fig = plt.figure(figsize=self.conf.get('figsize'))
            
            self.ax = Axes3D(self.fig) if self.axes3D else self.fig.gca()
        
        if 'grid' in self.conf.keys():
            self.ax.grid(b=self.conf['grid'], which='major')
        if 'minor_grid' in self.conf.keys():
            self.ax.grid(b=self.conf['minor_grid'], which='minor')
        
        if 'xlim' in self.conf.keys():
            self.ax.set_xlim(self.conf['xlim'][0], self.conf['xlim'][1])
        
        if 'ylim' in self.conf.keys():
            self.ax.set_ylim(self.conf['ylim'][0], self.conf['ylim'][1])
            
        if self.axes3D and 'zlim' in self.conf.keys():
            self.ax.set_zlim(self.conf['zlim'][0], self.conf['zlim'][1])
        
        if 'title' in self.conf.keys():
            self.ax.set_title(self.conf['title'])
            
        if 'xlabel' in self.conf.keys():
            self.ax.set_xlabel(self.conf['xlabel'])
        
        if 'ylabel' in self.conf.keys():
            self.ax.set_ylabel(self.conf['ylabel'])
        
        if self.axes3D and 'zlabel' in self.conf.keys():
            self.ax.set_zlabel(self.conf['zlabel'])
        
        if 'background' in self.conf.keys():
            self.ax.set_facecolor(self.conf['background'])
        
        if self.axes3D:
            self.ax.view_init(elev=self.conf.get('elevation', None), azim=self.conf.get('azimuth', None))


    def append(self, obj):
        
        self.scene_objects.append(obj)
        
        
    def set_xticks(self):
        
        if 'xticks' in self.conf.keys():
            
            if np.isscalar(self.conf['xticks']):
                lim = self.ax.get_xlim()
                tck = np.linspace(lim[0], lim[1], self.conf['xticks'])
            else:
                tck = self.conf['xticks']
                
            self.ax.set_xticks(tck)
            
        else:
            tck = self.ax.get_xticks()
            
            
        if 'xticklabels' in self.conf.keys():
            
            if isinstance(self.conf['xticklabels'], str):
                labels = [('{0:'+self.conf['xticklabels']+'}').format(s) for s in tck]
            else:
                labels = self.conf['xticklabels']
        
            self.ax.set_xticklabels(labels)
            
            
    def set_yticks(self):
        
        if 'yticks' in self.conf.keys():
            
            if np.isscalar(self.conf['yticks']):
                lim = self.ax.get_ylim()
                tck = np.linspace(lim[0], lim[1], self.conf['yticks'])
            else:
                tck = self.conf['yticks']
                
            self.ax.set_yticks(tck)
            
        else:
            tck = self.ax.get_yticks()
            
            
        if 'yticklabels' in self.conf.keys():
            
            if isinstance(self.conf['yticklabels'], str):
                labels = [('{0:'+self.conf['yticklabels']+'}').format(s) for s in tck]
            else:
                labels = self.conf['yticklabels']
        
            self.ax.set_yticklabels(labels)
            
            
    def set_zticks(self):
        
        if 'zticks' in self.conf.keys():
            
            if np.isscalar(self.conf['zticks']):
                lim = self.ax.get_zlim()
                tck = np.linspace(lim[0], lim[1], self.conf['zticks'])
            else:
                tck = self.conf['zticks']
                
            self.ax.set_zticks(tck)
            
        else:
            tck = self.ax.get_zticks()
            
            
        if 'zticklabels' in self.conf.keys():
            
            if isinstance(self.conf['zticklabels'], str):
                labels = [('{0:'+self.conf['zticklabels']+'}').format(s) for s in tck]
            else:
                labels = self.conf['zticklabels']
        
            self.ax.set_zticklabels(labels)
        
        
    def plot(self):
        
        for obj in sorted(self.scene_objects, key=lambda obj: obj.plot_last):
            
            if obj.visible: obj.draw_in_axes(self.ax)
                     
        self.set_xticks()
        self.set_yticks()
        if self.axes3D: self.set_zticks()
            
        labels = [None if not obj.visible else obj.opts.get('label', None) for obj in self.scene_objects]
            
        if any([l is not None for l in labels]) and not self.axes3D:
            self.ax.legend(loc=self.conf.get('legend_loc', 'best'))
    
    
    def show(self):
        
        self.plot()
        
        if 'tight_layout' in self.conf.keys():
            plt.tight_layout()
        
        plt.show()
        
        
        
class SceneGrid:
    
    def __init__(self, nrows, ncols, **kwargs):
        
        self.conf = kwargs.copy()
        
        self.nrows = nrows
        self.ncols = ncols
        
        self.gs = GridSpec(nrows, ncols)
        
        self.scenes = {}
        
        self.init()
        
        
    def init(self):
        
        self.fig = plt.figure(figsize=self.conf.get('figsize'))
        
        if 'title' in self.conf.keys():
            self.fig.suptitle(self.conf['title'])
        
    
    def add_scene(self, gspos, scene, **kwargs):
        
        if scene.axes3D:
            ax = self.fig.add_subplot(gspos, projection='3d')
        else:
            ax = self.fig.add_subplot(gspos)
            
        if scene.fig is not None:
            scene.fig.clear()
            
        s = scene.clone(deep=True)
        
        for k, v in kwargs.items():
            s.conf[k] = v
            
        s.ax = ax
        s.init()
        
        self.scenes[gspos] = s
        
    
    def plot(self):
        
        for scene in self.scenes.values():
            scene.plot()
        
        
    def show(self):
        
        self.plot()
            
        if 'tight_layout' in self.conf.keys():
            plt.tight_layout()
        
        plt.show()
        
        
        
class IsometricSceneGrid:
    
    def __init__(self, elevation, azimuth, scene, **kwargs):
        
        self.elevation = np.atleast_1d(elevation)
        self.azimuth = np.atleast_1d(azimuth)
        
        conf = kwargs.copy()
        
        if ('figsize' not in kwargs.keys()) and ('figsize' in scene.conf.keys()):
            conf['figsize'] = (len(self.elevation) * scene.conf['figsize'][0], len(self.azimuth) * scene.conf['figsize'][1])
        
        self.scene_grid = SceneGrid(len(self.elevation), len(self.azimuth), **conf)
        
        self.scene = scene
        
        for i, el in enumerate(self.elevation):
            for j, az in enumerate(self.azimuth):
                
                self.scene_grid.add_scene(self.scene_grid.gs[i,j], self.scene, elevation=el, azimuth=az)
        
    
    def plot(self):
        
        self.scene_grid.plot()
        
        
    def show(self):
        
        self.plot()
        
        plt.show()

        
        