import torch
import matplotlib.pyplot as plt


import torchvision
from utils.cifar import IndexedCifar10 as cifar10
from utils.mnist import IndexedMNIST as mnist

from IPython.display import clear_output
import time
import numpy as np
from scipy.stats.stats import spearmanr, pearsonr
import seaborn as sns
import pandas as pd



from functools import reduce
import operator

from utils.divergence import js_divergence, kl_divergence, js_distance,cartesian_js
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

dataset = mnist
dataset_name = 'mnist'

transform_test = dataset.transform_test

classes = dataset.classes()
MEAN = dataset.MEAN
STD = dataset.STD

trainset = dataset(transform_test, train=True)
testset = dataset(transform_test, train=False)

'''
from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save("./data/order_cifar_pie_test.npy", pie_order)

'''



def pie_name(num):
    return "p"+str(num)

RAW_PERCENTS = [0.0, 90.0]

PERCENTS = [pie_name(i) for i in RAW_PERCENTS]
EPOCHS = [0, 1]

LOSS_TYPE = 'mean'

############ learning dyanmic metric

def parallelGPU(model):
    return model #torch.nn.DataParallel(model)

def prod(iterable):
    return reduce(operator.mul, iterable, 1)
############# data


def collect_raw_sparsity_logs(files, log):
    start = time.time()
    runs = []
    for count, file in enumerate(files):    
        print ("Progress: {:0.2f} %, {} Seconds ".format((count + 1)/len(files) * 100, time.time() - start))
        clear_output(wait=True)

        run = {}
        
        with open(file + "/" + log, 'rb') as handle:
            run['train'] = pickle.load(handle)

        
        for percent in PERCENTS:
            with open(file + "/" + percent + "/" + log, 'rb') as handle:
                run[percent] = pickle.load(handle)
        
        runs.append(run)
    return runs

def collect_logs_per_sparsity(files, log, sparsity=None):
    start = time.time()
    runs = []
    for count, file in enumerate(files):    
        print ("Progress: {:0.2f} %, {} Seconds ".format((count + 1)/len(files) * 100, time.time() - start))
        clear_output(wait=True)

        
        with open(file + "/" + log, 'rb') as handle:
            run = pickle.load(handle)
            
            
        if sparsity is not None:
            with open(file + "/" + sparsity + "/" + log, 'rb') as handle:
                tune_run = pickle.load(handle)

            for i in range(0, len(run)):
                run[i]['hist'] += tune_run[i]['hist']
        
        
        '''for _, record in run.items():
            for i in record['hist']:
                i.pop('output')'''
            
            
        runs.append(run)
    return runs


def training_divergence(runs, epochs):
    '''
    return dict of examples
    '''
    examples = {}

    for run in runs:

        for idx, val in run.items():
            for epoch in epochs:

                if idx not in examples:
                    examples[idx] = {}
                if epoch not in examples[idx]:
                    examples[idx][epoch] = []
                examples[idx][epoch].append(val['hist'][epoch]['output'])

    agrs = {}

    pairs = [(e,epochs[-1]) for e in epochs]
    for p in pairs:
        agrs[p] = {}

    for idx, epochs in examples.items():
        print ("{0:.3f} % ".format(idx/len(examples) * 100))
        clear_output(wait=True)

        for p in pairs:
            agrs[p][idx] = cartesian_js(epochs[p[0]], epochs[p[1]])
    return agrs
  

############

def vardiff(x):
    xdiff = np.diff(a)
    if np.diff(a).mean() == 0:
        return 0
    else:
        return xdiff.std() / np.abs(xdiff.mean())

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))[0][1]

def first_learn(s, thresh):
    return np.argmax(s > thresh)

def spearmanr_corr(s):
    return spearmanr(s, list(range(0,len(s))), nan_policy='propagate')[0]

def pct_change(s):
    return np.mean(np.diff(s) / s[1:])


def getInvCount(arr): 
    n = len(arr)
    inv_count = 0
    for i in range(n): 
        for j in range(i + 1, n): 
            if (arr[i] > arr[j]): 
                inv_count += 1
    return inv_count 

def conseq_adj_inversion(s):
    ss = [int(elem) for elem in s]
    ss = np.ediff1d(ss)
    return (ss == -1).sum()

########### model divergence metric



def load_divergence_log(model_, prefix_, agr_, prune_):
    for i in EPOCHS:
        arr = np.load('./data/order_' + dataset_name + '_agr_' + str(i) + '_' + model_ + '_test.npy')
        agr_.append((prefix_ + str(i), arr))

    for i in PERCENTS:
        arr = np.load('./data/order_' + dataset_name + '_pie*' + i + '_' + model_ + '_test.npy')
        prune_.append((prefix_ + i, arr))


def plot_images(dataset, indexes, K, show_batch):
    fig, ax = plt.subplots(figsize=(15, 15))
    images = torch.stack([dataset[i][0] for i in indexes[:K]])
    show_batch(images)


def reverse_index(series):
    ''' 
    series are ranking of example indexes, but correlation requires ranks of each index
    '''
    x = np.zeros(len(series))
    for i,j in enumerate(series):
        x[j] = math.floor(i)
    return x

########################## VIZ

def show_cifar10_batch(images):
    images = make_cifar10_image(torchvision.utils.make_grid(images))
    plt.imshow(images)
    plt.show()
    
def make_cifar10_image(img):
    for i in range(0, 3):
        img[i] = img[i] * STD[i] + MEAN[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


def show_mnist_batch(images):
    im = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()

def make_mnist_image(img):
    img = img * STD + MEAN
    return img.numpy()

##########################

def convert_dict_to_rank(indexes, ascend=True):
    buf = np.array(list((indexes.values()))).argsort()
    return buf if ascend else buf[::-1]

def plot_acc(indexes, curr_runs):
    sz = len(indexes)
    res = []
    for run in curr_runs:
        accuracy = []
        train = run
        for epoch in range(0, len(train[0]['hist'])):
            acc = (np.sum([train[i]['hist'][epoch]['pred'] == train[i]['hist'][epoch]['target'] for i in indexes])) / sz
            accuracy.append(acc)
        res.append(accuracy)
    df = pd.concat([pd.DataFrame([(i,j) for i,j in enumerate(l)], columns=["epoch", "acc"]) for l in res])
    sns.lineplot(x="epoch", y="acc", data=df)

def plot_chunks(chunks, num, curr_runs):
    chunks_ = np.split(chunks, num)
    sns.set(color_codes=True)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    for c in chunks_:
        plot_acc(c, curr_runs)
    return ax

def reverse_index(series):
    ''' 
    series are ranking of example indexes, but correlation requires ranks of each index
    '''
    x = np.zeros(len(series))
    for i,j in enumerate(series):
        x[j] = math.floor(i)
    return x


########### PIE


def save(file_name, obj):
    with open(file_name + ".pickle", 'wb') as handle:
        pickle.dump(obj, handle)