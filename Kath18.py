import pdb
import torch
from train_config import send_data_to_device
import math
import collections
import numpy as np
import torch.nn.functional as F
            

r"""
Mostly copied from https://anonymous.4open.science/repository/c6d4060d-bdac-4d31-839e-8579650255b3/lib/trainer.py
"""

class Condition(object):
    """An interface for use with the ConditionalStartSampler."""
    @property
    def satisfied(self):
        raise NotImplementedError()

    @property
    def previously_satisfied(self):
        pass  # not necessary

    def update(self, scores):
        pass  # not necessary

class VarianceReductionCondition(Condition):
    """Sample with importance sampling when the variance reduction is larger
    than a threshold. The variance reduction units are in batch size increment.
    Arguments
    ---------
        vr_th: float
               When vr > vr_th start importance sampling
        momentum: float
                  The momentum to compute the exponential moving average of
                  vr
    """
    def __init__(self, vr_th=1.2, momentum=0.9):
        self._vr_th = vr_th
        self._vr = 0.0
        self._previous_vr = 0.0
        self._momentum = momentum

    @property
    def variance_reduction(self):
        return self._vr

    @property
    def satisfied(self):
        self._previous_vr = self._vr
        return self._vr > self._vr_th

    @property
    def previously_satisfied(self):
        return self._previous_vr > self._vr_th

    def update(self, scores):
        scores = np.array(scores)
        u = 1.0/len(scores)
        S = scores.sum()
        if S == 0:
            g = np.array(u)
        else:
            g = scores/S
        new_vr = 1.0 / np.sqrt(1 - ((g-u)**2).sum()/(g**2).sum())
        self._vr = (
            self._momentum * self._vr +
            (1-self._momentum) * new_vr
        )


class KathLossSelector(object):
    def __init__(self, size_to_backprops, pool_size, rank):
        
        assert (size_to_backprops < pool_size)
        
        self.size_to_backprops = size_to_backprops
        self.pool_size = pool_size
        
        tau_th = float(pool_size + 3*size_to_backprops) / (3*size_to_backprops)
        self.condition = VarianceReductionCondition(tau_th)
        
        self.rank = rank
        self.clear_pool()
        
    def init_for_this_epoch(self, epoch):
        pass
    
    
    def clear_pool(self):
        self.candidate_indexes = send_data_to_device(torch.LongTensor([]), self.rank)
        self.candidate_inputs = send_data_to_device(torch.Tensor([]), self.rank)
        self.candidate_targets = send_data_to_device(torch.LongTensor([]), self.rank)
            
    def get_probabilities(self, scores):
        return scores / scores.sum()
    
    def get_upweights(self, scores):
        return 1 / self.pool_size / scores
    
    def update_examples(self, model, criterion, inputs, targets, indexes, epoch):
        
        self.candidate_indexes = torch.cat((self.candidate_indexes, send_data_to_device(torch.LongTensor(indexes), self.rank)), 0)
        self.candidate_inputs = torch.cat((self.candidate_inputs, inputs), 0)
        self.candidate_targets = torch.cat((self.candidate_targets, targets), 0)
        
        if self.condition.satisfied:
            if len(self.candidate_indexes) >= self.pool_size:
                outputs = model(self.candidate_inputs)
                losses = criterion(outputs, self.candidate_targets)
                
                probs = self.get_probabilities(losses.detach().numpy())
                upweights = self.get_upweights(losses.detach().numpy())
                
                indices = np.random.choice(range(len(self.candidate_indexes)), self.size_to_backprops, replace=True, p=probs)
                
                inputs = self.candidate_inputs[indices]
                targets = self.candidate_targets[indices]
                indexes = self.candidate_indexes[indices]
                upweights = upweights[indices]
                
                self.condition.update(losses.detach().numpy())
                self.clear_pool()
                
                return inputs, targets, upweights, indexes
            
            else:
                return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
                
        else:
            if len(self.candidate_indexes) >= self.size_to_backprops:
                outputs = model(self.candidate_inputs)
                losses = criterion(outputs, self.candidate_targets)
        
                inputs = self.candidate_inputs
                targets = self.candidate_targets
                indexes = self.candidate_indexes
                
                self.condition.update(losses.detach().numpy())
                self.clear_pool()
                
                return inputs, targets, torch.ones(indexes.shape), indexes