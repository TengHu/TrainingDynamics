import pdb
import torch
from train_config import CONFIDENT_CORRECT_THRESH, CORRECT_THRESH, COMPACTION_STALNESS, COMPACTION_SAMPLE_PROB, WARMUP_EPOCH, send_data_to_device
import math
import collections
import numpy as np
from functools import partial

class ExampleCompactor(object):
    r"""
    Re-weight loss
    """
    def __init__(self,  rank):
        self.mask_history = collections.defaultdict(int)
        self.correct_history = collections.defaultdict(partial(collections.deque, maxlen=CORRECT_THRESH))
        
        
    def _use_stale(self, epoch):
        return COMPACTION_STALNESS != 0 and epoch % COMPACTION_STALNESS != 0
    
    def init_for_this_epoch(self, epoch):
        if not self._use_stale(epoch):
            self.mask_history.clear()
    
    
    def correct_compaction_consec(self, corrects_examples, epoch, loss, index, rank): 
        r"""
        Consecutive corrects
        """
        
        for i, idx in enumerate(index):
            self.correct_history[idx.item()].append(1 if corrects_examples[i].item() else 0)
        
        
        if self._use_stale(epoch):
            mask = np.array([self.mask_history[i.item()] for i in index])
        else:
            
            corrects = []
            for i in index:
                corrects += [sum(self.correct_history[i]) >= CORRECT_THRESH]
            mask = self._correct_compaction(np.array(corrects), epoch, loss, index, rank)
            for i, idx in enumerate(index):
                self.mask_history[idx.item()] = mask[i].item()
                
        return loss[mask], index[mask]
    
    
    def correct_compaction_confcorrect(self, confidence, corrects_examples, epoch, loss, index, rank): 
        r"""
        Confident Corrects
        """
        
        
        if self._use_stale(epoch):
            mask = np.array([self.mask_history[i.item()] for i in index])
        else:
            confidence_examples = (confidence >= CONFIDENT_CORRECT_THRESH)
            mask = self._correct_compaction(corrects_examples * confidence_examples, epoch, loss, index, rank)
            
            if COMPACTION_STALNESS > 0:
                for i, idx in enumerate(index):
                    self.mask_history[idx.item()] = mask[i].item()
                
        return loss[mask], index[mask]
    
    
    
    
    def _correct_compaction(self, corrects_examples, epoch, loss, index, rank):     
        mask = send_data_to_device(torch.ones(corrects_examples.shape) == 1, rank)
        if epoch >= WARMUP_EPOCH:

            # 1) scale all confident example losses
            loss[corrects_examples] *=  (1 / COMPACTION_SAMPLE_PROB)

            # 2) compact confident examples
            # a. compute random mask
            random_mask = send_data_to_device(torch.rand(len(mask)),rank) < COMPACTION_SAMPLE_PROB

            # b. sample the confident ones
            mask[corrects_examples] = random_mask[corrects_examples]
        return mask
    
    ######
    
    '''def confidence_compaction(self, confidence, epoch, loss, index, rank):  
        if self._use_stale(epoch):
            mask = np.array([self.mask_history[i.item()] for i in indexes])
        else:
            mask = self._confidence_compaction(confidence, epoch, loss, index, rank)
        return loss[mask], index[mask]
        
    def _confidence_compaction(self, confidence, epoch, loss, index, rank):
        mask = send_data_to_device(torch.ones(confidence.shape) == 1, rank)
        if epoch >= WARMUP_EPOCH:
            # get confidence examples
            confidence_examples = (confidence >= CONFIDENCE_COMPACTION_CONFIDENCE_THRESH)

            # control
            if RANDOM_CONTROL:
                fake_confidence_mask = np.random.choice(range(len(inputs)), confidence_examples.sum().item(), replace=False)
                fake_confidence_examples = np.zeros(len(inputs)) 
                fake_confidence_examples[fake_confidence_mask] = 1
                confidence_examples = fake_confidence_examples == 1

            # 1) scale all confident example losses
            loss[confidence_examples] *=  (1 / COMPACTION_SAMPLE_PROB)

            # 2) compact confident examples
            # a. compute random mask
            random_mask = send_data_to_device(torch.rand(len(mask)),rank) < COMPACTION_SAMPLE_PROB

            # b. sample the confident ones
            mask[confidence_examples] = random_mask[confidence_examples]
        return mask
    '''