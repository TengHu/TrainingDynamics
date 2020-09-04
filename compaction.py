import pdb
import torch
from train_config import CONFIDENT_CORRECT_THRESH, COMPACTION_STALNESS, COMPACTION_SAMPLE_PROB, WARMUP_EPOCH, send_data_to_device
import math
import collections
import numpy as np
from functools import partial

class CompactionSelector(object):
    def __init__(self,  rank):
        self.select_history = collections.defaultdict(int)
        
    def _use_stale(self, epoch):
        return COMPACTION_STALNESS != 0 and epoch % COMPACTION_STALNESS != 0
    
    def init_for_this_epoch(self, epoch):
        if not self._use_stale(epoch):
            self.select_history.clear()
    
    
    def confcorrect_compaction(self, confidence, corrects_examples, epoch, loss, index, rank): 
        if self._use_stale(epoch):
            selected = np.array([self.select_history[i.item()] for i in index])
        else:
            confidence_examples = (confidence >= CONFIDENT_CORRECT_THRESH)
            selected = corrects_examples * confidence_examples
            
            if COMPACTION_STALNESS > 0:
                for i, idx in enumerate(index):
                    self.select_history[idx.item()] = selected[i].item()

        mask = self._compact_and_upweight_loss(selected, epoch, loss,   rank)
        return loss[mask], index[mask]
    
    def _compact_and_upweight_loss(self, selected, epoch, loss, rank):     
        mask = send_data_to_device(torch.ones(selected.shape) == 1, rank)
        if epoch >= WARMUP_EPOCH:
            # 1) upweight all selected example losses
            loss[selected] *=  (1 / COMPACTION_SAMPLE_PROB)

            # 2) compute random mask
            random_mask = send_data_to_device(torch.rand(len(mask)),rank) < COMPACTION_SAMPLE_PROB

            # 3) sample the confident ones
            mask[selected] = random_mask[selected]
        return mask