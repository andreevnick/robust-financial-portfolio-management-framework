# Copyright 2021 portfolio-robustfpm-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

from numpy import asarray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

__all__ = [
          'plot_surface3D',
          'plot_points3D'
          ]

def plot_surface3D(Xgrid, Ygrid, Zgrid, ax = None, figsize=(12,10), **kwargs):
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
    
    if len(Zgrid) == 0:
        return ax
    
    ax.plot_surface(Xgrid, Ygrid, Zgrid, **kwargs)
    
    return ax

def plot_points3D(x, y, z, ax = None, figsize=(12,10), **kwargs):
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
    
    x = asarray(x)
    y = asarray(y)
    z = asarray(z)
    
    if z.size == 0:
        return ax
    
    ax.scatter(x, y, z, **kwargs)
    
    return ax