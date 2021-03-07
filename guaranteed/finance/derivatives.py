# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

import numpy as np


__all__ = [
            'IOption',
            'Option',
            'PutOnMaxOption',
            'CallOnMaxOption',
            'Put2Call1Option'
          ]



class IOption:
    
    def payoff(self, S):
        
        raise Exception("The method must be defined in a subclass")


        
class Option(IOption):
    """ Option Factory.
    """
    
    def __init__(self, option_type, **kwargs):
        
        if option_type.lower() == 'putonmax':
            
            self.option = PutOnMaxOption(kwargs.pop('strike'))
            
        elif option_type.lower() == 'callonmax':
            
            self.option = CallOnMaxOption(kwargs.pop('strike'))
            
        elif option_type.lower() == 'put2call1':
            
            self.option = Put2Call1Option()
            
    
    def payoff(self, S):
        
        return self.option.payoff(S)
    


class PutOnMaxOption:
    
    
    def __init__(self, K):
        
        self.K = K


    def payoff(self, S):

        return np.maximum(self.K - S.max(axis=1), float(0))
    
    
class CallOnMaxOption:
    
    
    def __init__(self, K):
        
        self.K = K


    def payoff(self, S):

        return np.maximum(S.max(axis=1) - self.K, float(0))
    

class Put2Call1Option:
    
    
    def __init__(self):

        pass

    def payoff(self, S):

        return np.maximum(S[:,0] - S[:,1], float(0))
    
    