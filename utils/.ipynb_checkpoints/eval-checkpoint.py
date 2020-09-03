from __future__ import print_function, absolute_import


import numpy as np


__all__ = ['compute_pred']

def compute_pred(targets, outputs, topk=(1,)):
    maxk = max(topk)
    _, pred = outputs.data.topk(maxk, 1, True, True)
    pred = pred.t()
    batch_size = targets.data.size(0)
    correct = pred.eq(targets.data.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return pred, res
