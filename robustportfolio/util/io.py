# Copyright 2021 portfolio-robustportfolio-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import os


__all__ = [
          'fig2files'
          ]
          
          
def fig2files(plt, dirname, filename, dpi=None):
    
    os.makedirs('{0}/png'.format(dirname), exist_ok=True)
    plt.savefig('{0}/png/{1}.png'.format(dirname, filename), dpi=dpi)
    
    os.makedirs('{0}/eps'.format(dirname), exist_ok=True)
    plt.savefig('{0}/eps/{1}.eps'.format(dirname, filename), format='eps')
    
    os.makedirs('{0}/svg'.format(dirname), exist_ok=True)
    plt.savefig('{0}/svg/{1}.svg'.format(dirname, filename), format='svg')
          