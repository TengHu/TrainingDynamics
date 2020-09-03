from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from collections import defaultdict

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
__all__ = ['Logger']


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.fpath = fpath
        self.log = None
        self.blob = defaultdict(list)
        
        
        
        
        self.resume = resume
        self.title = '' if title == None else title
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

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

        
        

    
    def set_example_log(self, dataset):
        self.log = {}
        classes = dataset.classes()
        for i in iter(dataset):
            self.log[i[2]] = {'class': classes[i[1]], 'hist': []}
            
        
    def set_metadata(self, model, args):
        pass

    def __getitem__ (self, index):
        return self.log[index]['hist']
    

    '''def update_train_log (self, index, targets, outputs, pred):
        res = targets == pred[0]  # top-1
        for idx, val in enumerate(index):
            self.log[val.item()]['hist'].append({
                "target": targets[idx].item(), "correct": res[idx].item(), "pred": pred[0][idx].item(),
                "output": outputs[idx].tolist()
            })

    def update_test_log(self, index, targets, outputs, pred):
        res = targets == pred[0] # top-1
        for idx, val in enumerate(index):
            self.log[val.item()]['hist'].append({
                "target": targets[idx].item(), "correct":res[idx].item(), "pred": pred[0][idx].item(),
                "output":outputs[idx].tolist()
            })'''
            
            
    

    def close(self):
        # pickle
        if len(self.blob) > 0:
            with open(self.fpath + '.logger.blob.pickle', 'wb') as handle:
                pickle.dump(self.blob, handle, pickle.HIGHEST_PROTOCOL)
        
        if self.log:
            with open(self.fpath + '.logger.pickle', 'wb') as handle:
                pickle.dump(self.log, handle, pickle.HIGHEST_PROTOCOL)

        if self.file is not None:
            self.file.close()

                    
if __name__ == '__main__':
    pass
