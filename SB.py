import pdb
import torch
from train_config import send_data_to_device
import math
import collections
import numpy as np
import torch.nn.functional as F
            

r"""
Mostly copied from https://anonymous.4open.science/repository/c6d4060d-bdac-4d31-839e-8579650255b3/lib/calculators.py
"""
class BoundedHistogram(object):
    r"""
    Histogram to approximate CDF
    """
    def __init__(self, max_history):
        self.max_history =  max_history
        self.history = collections.deque(maxlen=self.max_history)

    def append(self, value):
        self.history.append(value)

    # CDF
    def get_cdf_percentile(self, it, score):
        count = 0
        for i in it:
            if i < score:
                count += 1
        return count

    def percentile_of_score(self, score):
        num_lower_scores = self.get_cdf_percentile(self.history, score)
        return num_lower_scores * 100. / len(self.history)
    
    

class BatchedRelativeProbabilityCalculator(object):
    r"""
    Take losses, return probability
    """
    def __init__(self, history_length, beta, sampling_min=0):
        self.historical_losses = BoundedHistogram(history_length)         
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
        
        
        # disentangle the shape and prob
        draw = np.random.uniform(0, 1, size=select_probability.shape)
        return draw < select_probability, select_probability
       
    
class SBSelector(object):
    def __init__(self, size_to_backprops, beta, history, floor, mode, staleness, rank, target=0):
        self.size_to_backprops = size_to_backprops
        
        self.candidate_indexes = send_data_to_device(torch.LongTensor([]), rank)
        self.candidate_inputs = send_data_to_device(torch.Tensor([]), rank)
        self.candidate_targets = send_data_to_device(torch.LongTensor([]), rank)
        self.candidate_upweights = send_data_to_device(torch.Tensor([]), rank)
        self.rank = rank
        
        self.mode = mode
        self.staleness = staleness
        
        self.mask_calculator = BatchedRelativeProbabilityCalculator(history, beta, floor)
        self.stale_loss = collections.defaultdict()
        
        self.target = target
        
    
    def _update(self, inputs, targets, mask, indexes, upweights):
        self.candidate_indexes = torch.cat((self.candidate_indexes, send_data_to_device(torch.LongTensor(indexes[mask]), self.rank)), 0)
        self.candidate_inputs = torch.cat((self.candidate_inputs, inputs[mask]), 0)
        self.candidate_targets = torch.cat((self.candidate_targets, targets[mask]), 0)
        self.candidate_upweights = torch.cat((self.candidate_upweights, send_data_to_device(torch.Tensor(upweights[mask]), self.rank)), 0)

        
        if len(self.candidate_inputs) >= self.size_to_backprops:
            new_indexes = self.candidate_indexes[:self.size_to_backprops]
            new_inputs = self.candidate_inputs[:self.size_to_backprops] #.clone()
            new_targets = self.candidate_targets[:self.size_to_backprops] #.clone()
            new_upweights = self.candidate_upweights[:self.size_to_backprops]
            
            self.candidate_indexes = self.candidate_indexes[self.size_to_backprops:]
            self.candidate_inputs = self.candidate_inputs[self.size_to_backprops:]
            self.candidate_targets = self.candidate_targets[self.size_to_backprops:]
            self.candidate_upweights = self.candidate_upweights[self.size_to_backprops:]

            return new_inputs, new_targets, 1 / new_upweights, new_indexes
        else: 
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        
        
    def _update1(self, inputs, targets, mask, indexes, upweights):
        return inputs[mask], targets[mask],  send_data_to_device(torch.Tensor(1 / upweights[mask]), self.rank), indexes[mask]
    
    def _update_from_stale(self, inputs, targets, indexes):
        mask, weights = self.mask_calculator.select(np.array([self.stale_loss[i.item()] for i in indexes]))
        return self._update(inputs, targets, mask, indexes, weights)
    
    def _update_fresh(self, inputs, targets, losses, indexes):
        mask, weights = self.mask_calculator.select(losses.detach().cpu().numpy())

        # save losses
        if self.staleness > 0:
            for i, idx in enumerate(index):
                self.stale_loss[idx.item()] = losses[i].item()
        
        return self._update(inputs, targets, mask, indexes, weights)
            
    def _use_stale(self, epoch):
        return self.staleness != 0 and epoch % self.staleness != 0
    
    def init_for_this_epoch(self, epoch):
        if not self._use_stale(epoch):
            self.stale_loss.clear()
            
            
    def update_examples(self, model, criterion, inputs, targets, indexes, epoch):
        if self.mode == 0:
            return self.update_examples_with_loss(model, criterion, inputs, targets, indexes, epoch)
        elif self.mode == 1:
            return self.update_examples_with_grad_norm(model, criterion, inputs, targets, indexes, epoch)
        elif self.mode == 2:
            return self.update_examples_with_loss(model, criterion, inputs, targets, indexes, epoch)
        elif self.mode == 3:
            return self.update_examples_with_entropy(model, criterion, inputs, targets, indexes, epoch)
    
    
    def _entropy(self,tensor):
        return (tensor.log2() * -tensor).sum(dim=-1)
        
    
    def update_examples_with_entropy(self, model, criterion, inputs, targets, indexes, epoch):
        if self._use_stale(epoch):
            return self._update_from_stale(inputs, targets, indexes)
        else:
            # Selection pass
            outputs = model(inputs)
            
            preds = F.softmax(outputs, dim=-1)
            entropies = self._entropy(preds)
            
            return self._update_fresh(inputs, targets, entropies, indexes)
    
    def update_examples_with_loss(self, model, criterion, inputs, targets, indexes, epoch):
        if self._use_stale(epoch):
            return self._update_from_stale(inputs, targets, indexes)
        else:
            # Selection pass
            outputs = model(inputs)
            
            if self.target == 0:
                loss = criterion(outputs, targets)
            elif self.target == 1:
                new_targets = outputs.topk(k=1, dim=-1)[1][:,-1]
                loss = criterion(outputs, new_targets)
            
            if self.mode == 2:
                loss = send_data_to_device(torch.rand(loss.shape), self.rank)
            
            return self._update_fresh(inputs, targets, loss, indexes)
        
    def update_examples_with_grad_norm(self, model, criterion, inputs, targets, indexes, epoch):
        
        #### Compute grad
        def _capture_activations(layer, input, output):
            setattr(layer, "activations", input[0].detach())
        
        def _capture_backprops(layer, _input, output):

            assert not hasattr(layer, 'backprops_list'), "Seeing result of previous backprop, use clear_backprops(model) to clear"
            if not hasattr(layer, 'backprops_list'):
                setattr(layer, 'backprops_list', [])
            layer.backprops_list.append(output[0].detach())

        
        last_layer = torch.nn.Sequential(*(list(model.modules())))[-1]
        
        handles = []
        handles.append(last_layer.register_forward_hook(_capture_activations))
        handles.append(last_layer.register_backward_hook(_capture_backprops))
        
         
        # model.fc3.backprops_list
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        
        loss.mean().backward()
        
        A = last_layer.activations
        n = A.shape[0]
        B = last_layer.backprops_list[0] * n
        og = torch.einsum('ni,nj->nij', B, A)
        
        scores = og.norm(p=2, dim=(1,2))
        
        #### Clear up
        del last_layer.backprops_list
        del last_layer.activations
        
        for handle in handles:
            handle.remove()
        
        return self._update_fresh(inputs, targets, scores, indexes)
