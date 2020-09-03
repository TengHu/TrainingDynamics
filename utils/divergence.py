import torch
import torch.nn.functional as F
import numpy as np

import math

__all__ = ['js_distance', 'js_divergence', 'softmax_jsd', 'cartesian_js']


def kl_divergence(p, q):
    EPSILON = 1e-8
    ''' Pytorch Bug: https://github.com/pytorch/pytorch/issues/32520 '''
    return max(EPSILON, F.kl_div(q.log(), p, None, None, "sum"))
    #return sum(p[i] * (p[i]/q[i]).log() if p[i] > 0 else 0 for i in range(len(p)))
    
    
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def js_distance(p, q):
    return np.sqrt(js_divergence(p, q))


def softmax_jsd(p, q, dim):
    p, q = F.softmax(torch.Tensor(p), dim=dim), F.softmax(torch.Tensor(q), dim=dim)
    return js_divergence(p,q)


def cartesian_js(p, q, dim=0):
    return np.mean([softmax_jsd(pp, qq, dim) for pp in p for qq in q])


def check(num, target):
    eps = 1e-3
    assert(abs(num - target)  < eps)

if __name__ == '__main__':
    
    # natural log
    p = torch.Tensor([0.10, 0.40, 0.50])
    q = torch.Tensor([0.80, 0.15, 0.05])
    check(kl_divergence(p,q), 1.3357)
    check(kl_divergence(q,p), 1.4013)
    
    check(js_divergence(p,q), 0.2913)
    check(js_divergence(q,p), 0.2913)
    
    
    p = torch.Tensor([1.0, 0.0])
    q = torch.Tensor([0.5, 0.5])
    check(js_distance(p,q), 0.4645)
    check(js_distance(q,p), 0.4645)
    
    