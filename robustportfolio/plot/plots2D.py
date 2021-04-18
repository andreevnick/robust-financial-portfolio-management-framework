# Copyright 2021 portfolio-robustportfolio-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import matplotlib.pyplot as plt 

import seaborn as sns

import numpy as np


__all__ = [
          'plot_points2D'
          ]


def plot_points2D(point_array, padding = 0.1, style = 'ok', figsize=(10,8), ax = None, **kwargs):
    
    if ax is None:
        ax = plt.figure(figsize=figsize).gca()
        
    if len(point_array) == 0:
        return ax
    
    data = np.asarray(point_array)
    
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    
    ax.plot(data[:,0], data[:,1], style, **kwargs)
    
    if data_max[0] + padding * (data_max[0]-data_min[0]) - (data_min[0] - padding * (data_max[0]-data_min[0])) > 1e-4:
        ax.set_xlim(data_min[0] - padding * (data_max[0]-data_min[0]), data_max[0] + padding * (data_max[0]-data_min[0]))
        
    if data_max[1] + padding * (data_max[1]-data_min[1]) - (data_min[1] - padding * (data_max[1]-data_min[1])) > 1e-4:
        ax.set_ylim(data_min[1] - padding * (data_max[1]-data_min[1]), data_max[1] + padding * (data_max[1]-data_min[1]))
        
    return ax
    