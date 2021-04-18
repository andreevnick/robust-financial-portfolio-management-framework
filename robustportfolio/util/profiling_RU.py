# Copyright 2021 portfolio-robustportfolio-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

from datetime import datetime
import time

from .util import coalesce, keydefaultdict


__all__ = [
            'Timer',
            'PTimer',
            'ProfilerData'
          ]


class Timer:
    """ Console profiler decorated as Python context manager. Displays the elapsed time
    of the block after execution.
    
    Parameters
    ----------
    header : string
        Displayed header for the executed block.
    flush : boolean
        When True, the header is displayed before the execution starts. Otherwise,
        the header is printed after execution with the elapsed time part. Default is False.
    silent: boolean
        When True, no information is displayed in the console. Default is False.
    
    """
   

    def __init__(self, header=None, flush=False, silent=False):
        
        self.header = coalesce(header, 'Время расчетов')
        self.flush = flush
        self.silent = silent
            

    def start(self):
        
        self.__start = datetime.now()
        self.__start_cpu = time.perf_counter()
        
        
    def stop(self):
        
        self.__end_cpu = time.perf_counter()
        
        self.elapsed_sec = (datetime.now() - self.__start).total_seconds()
        self.elapsed_sec_cpu = self.__end_cpu - self.__start_cpu
        
        self.elapsed_hms = Timer.sec2hms(self.elapsed_sec)
        self.elapsed_hms_cpu = Timer.sec2hms(self.elapsed_sec_cpu)
        

    def __enter__(self):
        
        if self.flush and not self.silent:
            print(self.header, end=': ')
        
        self.start()
        
        return self
    

    def __exit__(self, *args):
        
        self.stop()

        if not self.flush and not self.silent:
            print(self.header, end=': ')
        
        if not self.silent:
            print('{1} (CPU {2})'.format(self.header, Timer.fancystr(self.elapsed_hms), Timer.fancystr(self.elapsed_hms_cpu)))
        
        
    @staticmethod
    def sec2hms(sec):
        
        mnt = int(sec // 60)
        sec = sec - 60 * mnt
        
        hr = int(mnt // 60)
        mnt = int(mnt - hr * 60)
        
        return (hr, mnt, sec)
    
        
    @staticmethod
    def fancystr(hms):
    
        if not isinstance(hms, tuple): # hms is total sec
            hms = Timer.sec2hms(hms)
        
        res = ''
        
        if hms[0] > 0: res += '{0} ч '.format(hms[0])
        if (res != '') or (hms[1] > 0): res += '{0} мин '.format(hms[1])
        if ((res != '') and (hms[2] > 0)) or (res == ''): res += '{0:.4f} сек'.format(hms[2])
            
        return res
            
            
    def __str__(self):
        
        return '{0:02d}:{1:02d}:{2:02.4f}'.format(*self.elapsed_hms)

    
    
class ProfilerData:
    """ Utility class for the Timer class.
    
    """
    
    def __init__(self, header=None):
        
        self.header = coalesce(header, 'Суммарное время')
        
        self.data = keydefaultdict(ProfilerData)
        
    
    def total_sec(self):
        
        s = 0
        
        for k,v in self.data.items():
            
            if isinstance(v, ProfilerData):
                s += v.total_sec()
            else:
                s += v
                
        return s
    
    
    def print(self, indent=''):
        
        print('{0}{1}: {2}'.format(indent, self.header, Timer.fancystr(self.total_sec())))
        
        for k,v in self.data.items():
            
            if isinstance(v, ProfilerData):
                v.print(indent = indent + '|- ')
            else:
                print('{0}{1}: {2}'.format(indent + '|- ', k, Timer.fancystr(v)))
        
        
class PTimer(Timer):
    """ Extension of the Timer class. Allows to pass down the profiler data and append to it
    to reuse it in another profiler.
    
    Parameters
    ----------
    profiler_data : ProfilerData object
        Saved information about the previously executed blocks.
        
    See also
    ----------
    Timer
        
    """
    
    def __init__(self, header=None, flush=False, silent=False, profiler_data=None):
        
        super().__init__(header=header, flush=flush, silent=silent)
        
        self.profiler_data = coalesce(profiler_data, ProfilerData())
        
        
    def __enter__(self):
        
        self.profiler_data.data[self.header] = self.profiler_data.data.get(self.header, float(0))
        
        return super().__enter__()
        
        
    def __exit__(self, *args):
        
        super().__exit__(args)
        
#         self.profiler_data.data[self.header] = self.profiler_data.data.get(self.header, float(0)) + self.elapsed_sec

        self.profiler_data.data[self.header] += self.elapsed_sec
    