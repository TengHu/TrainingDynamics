import pdb
import torch
from train_config import SB_BETA, SB_HISTORY_SIZE, SB_STALNESS, send_data_to_device
import math
import collections
import numpy as np


class UnboundedHistogram(object):
    r"""
    Stateful Histogram approximate CDF
    """
    def __init__(self, max_history):
        self.max_history =  max_history
        self.history = collections.deque(maxlen=self.max_history)

    def append(self, value):
        self.history.append(value)

    def get_count(self, it, score):
        count = 0
        for i in it:
            if i < score:
                count += 1
        return count

    def percentile_of_score(self, score):
        num_lower_scores = self.get_count(self.history, score)
        return num_lower_scores * 100. / len(self.history)
    
class BatchedRelativeProbabilityCalculator(object):
    r"""
    Pass losses return probabilities
    """
    def __init__(self, history_length, beta, sampling_min=0):
        self.historical_losses = UnboundedHistogram(history_length)         
        self.sampling_min = sampling_min
        self.beta = beta

    def _update_history(self, losses):
        for loss in losses:
            self.historical_losses.append(loss)

    def _calculate_probability(self, loss):
        percentile = self.historical_losses.percentile_of_score(loss)
        return math.pow(percentile / 100., self.beta)

    def _get_probability(self, losses):
        self._update_history(losses)
        probs = np.array([max(self.sampling_min, self._calculate_probability(loss)) for loss in losses])
        return probs

    def select(self, losses):
        select_probability = self._get_probability(losses)
        draw = np.random.uniform(0, 1, size=select_probability.shape)
        return draw < select_probability
        
    
    
class ExampleSelector(object):
    r"""
    Filter inputs/targets based on losses
    
    preserve loss history
    """
    def __init__(self, size_to_backprops, rank):
        self.size_to_backprops = size_to_backprops
        self.num_selected_for_bp = 0
        self.epoch_counter = 0
        
        self.candidate_inputs = send_data_to_device(torch.Tensor([]), rank)
        self.candidate_targets = send_data_to_device(torch.LongTensor([]), rank)
        
        self.mask_calculator = BatchedRelativeProbabilityCalculator(SB_HISTORY_SIZE, SB_BETA)
        
        self.stale_loss = collections.defaultdict()
        
    
    def _update(self, inputs, targets, mask):
        self.candidate_inputs = torch.cat((self.candidate_inputs, inputs[mask]), 0)
        self.candidate_targets = torch.cat((self.candidate_targets, targets[mask]), 0)

        if len(self.candidate_inputs) >= self.size_to_backprops:
            new_inputs = self.candidate_inputs[:self.size_to_backprops] #.clone()
            new_targets = self.candidate_targets[:self.size_to_backprops] #.clone()

            self.candidate_inputs = self.candidate_inputs[self.size_to_backprops:]
            self.candidate_targets = self.candidate_targets[self.size_to_backprops:]

            self.num_selected_for_bp += self.size_to_backprops

            return new_inputs,new_targets
        else: 
            return torch.empty(0), torch.empty(0)
    
    def _update_from_stale(self, inputs, targets, indexes):
        mask = self.mask_calculator.select(np.array([self.stale_loss[i.item()] for i in indexes]))
        return self._update(inputs, targets, mask)
    
    def _update_fresh(self, inputs, targets, losses, index):
        mask = self.mask_calculator.select(losses.detach().cpu().numpy())

        # save losses
        if SB_STALNESS > 0:
            for i, idx in enumerate(index):
                self.stale_loss[idx.item()] = losses[i].item()
        return self._update(inputs, targets, mask)
            
    def _use_stale(self, epoch):
        return SB_STALNESS != 0 and epoch % SB_STALNESS != 0
    
    def init_for_this_epoch(self, epoch):
        if not self._use_stale(epoch):
            self.stale_loss.clear()
    
    def update_examples(self, model, criterion, inputs, targets, indexes, losses, epoch):
        if self._use_stale(epoch):
            return self._update_from_stale(inputs, targets, indexes)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            return self._update_fresh(inputs, targets, loss, indexes)
