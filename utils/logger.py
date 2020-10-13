from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np
import sys

import time
from collections import defaultdict

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_config import LOG_TO_DISK

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
__all__ = ['Logger']

r"""
feature:
periodically dump blob ? 
or periodically append data to blob ? seems easier
"""

def _do_nothing(func):
    def wrapper(*args, **kwargs):
        pass
    return func if LOG_TO_DISK else wrapper

class Logger(object):
    @_do_nothing
    def __init__(self, fpath, resume=False): 
        self.file = None
        self.fpath = fpath
        
        # append only
        self.blob = defaultdict(list)
        
        self.resume = resume
        
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    @_do_nothing
    def set_names(self, names):
        if self.resume: 
            pass
        
        # initialize numbers as empty list
        # self.numbers is history of training
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
            
        self.file.write('\n')
        self.file.flush()

        
    @_do_nothing
    def append_blob(self, message):
        self.file.write(message)
        self.file.write('\n')
        self.file.flush()
        
    @_do_nothing
    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()
    
    @_do_nothing
    def dump_blob(self):
        if len(self.blob) > 0:
            with open(self.fpath + '.logger.blob.pickle', 'wb') as handle:
                pickle.dump(self.blob, handle, pickle.HIGHEST_PROTOCOL)
    
    @_do_nothing
    def close(self):
        self.dump_blob()
        if self.file is not None:
            self.file.close()
      
                    
if __name__ == '__main__':
    test_logger = Logger("a")
    pass
