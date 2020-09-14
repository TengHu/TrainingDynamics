from __future__ import print_function

import argparse
import os
import time
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from models.model_factory import create_model
from utils import Bar, Logger
from utils.misc import *
from utils.eval import *
import shutil
import torch.distributed as dist
from utils.mnist import IndexedMNIST
from utils.cifar import IndexedCifar10
import random
from train_config import *
from reweighter import SBSelector
from compaction import CompactionSelector

#from scheduler import BackpropsMultiStepLR
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def save_checkpoint(state, is_best, checkpoint, filename='finetuned.pth.tar'):
    if LOG_TO_DISK:
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def get_dataset(state):
    if state['dataset'] == 'cifar10':
        dataloader = IndexedCifar10
        num_classes = 10
    elif state['dataset'] == 'mnist':
        dataloader = IndexedMNIST
        num_classes = 10
    elif state['dataset'] == 'fashion':
        dataloader = IndexedFashion
        num_classes = 10
    else:
        raise Exception("unrecognized dataset")
    return dataloader, num_classes
   
def arguments():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    
    parser.add_argument('-d', '--dataset', default='mnist', type=str)
    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument(
        '--epochs',
        default=100,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)')
    parser.add_argument(
        '--train-batch',
        default=64,
        type=int,
        metavar='N',
        help='train batchsize')
    parser.add_argument(
        '--test-batch',
        default=50,
        type=int,
        metavar='N',
        help='test batchsize')
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.1,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        '--drop',
        '--dropout',
        default=0,
        type=float,
        metavar='Dropout',
        help='Dropout ratio')
    
    parser.add_argument(
        '--schedule',
        type=int,
        nargs='*',
        default=[],
        help='Decrease learning rate at these epochs.')
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.')
    parser.add_argument(
        '--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument(
        '--weight-decay',
        '--wd',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)')
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)')

    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='fcnet')
    parser.add_argument('--depth', type=int, default=8, help='Model depth.')
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    save_dir = './result-' + uuid.uuid4().hex
    parser.add_argument('--save_dir', default=save_dir + '/', type=str)

    return parser.parse_args()


def _main(rank=0):
    args = arguments()
    state = {k: v for k, v in args._get_kwargs()}
    set_random_seed(state)
    print(state)
    run(rank, state)
    
    
def main(rank=0):
    maybe_init_process_group(rank, _main)

    
def run(rank, state):
    global global_step
    global best_acc
    global train_logger
    global test_logger
    global num_classes
    
    best_acc = 0
    global_step = 0
    
    start_epoch = state['start_epoch']

    if LOG_TO_DISK and (not os.path.isdir(state['save_dir'])):
        mkdir_p(state['save_dir'])

    dataloader, num_classes = get_dataset(state)

    transform_train = dataloader.transform_train
    transform_test = dataloader.transform_test

    # Data
    print('==> Preparing dataset %s' % state['dataset'])
    trainset = dataloader(
        root='./data', train=True, download=True,transform=transform_train)
    trainloader = data.DataLoader(
        trainset,
        batch_size=state['train_batch'],
        shuffle=True,
        pin_memory=True,
        num_workers=state['workers'])
    
    

    testset = dataloader(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(
        testset,
        batch_size=state['test_batch'],
        shuffle=True,
        pin_memory=True,
        num_workers=state['workers'])
    

    # Model
    print("==> creating model '{}'".format(state['arch']))

    model = create_model(state, num_classes)
    model = send_model_to_device(model, rank)
    

    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    print('Total params: %.2fM' % (num_params))

    criterion = send_data_to_device(nn.CrossEntropyLoss(reduction='none'), rank)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=state['lr'],
        momentum=state['momentum'],
        nesterov=True,
        weight_decay=state['weight_decay'])
    
    r"""
    We use scheduler to anneal learning rate based on number of BACKPROPS.
    Look https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/lr_scheduler.py#L394 for implementations.
    """
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=state['schedule'], gamma=state['gamma'])
    
    title = state['arch']

    if state['resume']:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            state['resume']), 'Error: no checkpoint directory found!'
        state['save_dir'] = os.path.dirname(state['resume'])
        checkpoint = torch.load(state['resume'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        train_logger = Logger(
            os.path.join(state['save_dir'], 'train-log.txt'),
            resume=True)
        test_logger = Logger(
            os.path.join(state['save_dir'], 'test-log.txt'),
            resume=True)
    else:
        train_logger = Logger(
            os.path.join(state['save_dir'], 'train-log.txt'))
        
        
        
    train_logger.append_blob("model: {}, num_params: {}, lr: {}, weight_decay: {}, momentum:{}, batch size: {}, epoch: {}, seed: {}".format(state['arch'], num_params, state['lr'], state['weight_decay'], state['momentum'], state['train_batch'], state['epochs'], state['manualSeed']))

    if CORRECT_COMPACTION:
        train_logger.append_blob("correct compaction: {}, sample prob: {}, epoch: {}, stale: {}, confidence {}".format(CORRECT_COMPACTION, COMPACTION_SAMPLE_PROB, WARMUP_EPOCH, COMPACTION_STALNESS, CONFIDENT_CORRECT_THRESH))

    if SELECTIVE_BACKPROP:
        train_logger.append_blob("selective backprops on, beta {}, history size {}, staleness {}, warmup epoch {}, prob floor {}".format(SB_BETA, SB_HISTORY_SIZE, SB_STALNESS, SB_WARMUP_EPOCH, PROB_FLOOR))           
    
    if SELECTIVE_BACKPROP:
        selector = SBSelector(trainloader.batch_size, rank)
    elif CORRECT_COMPACTION:
        selector = CompactionSelector(rank)
    else:
        selector = None
    
    train_logger.set_names([
        'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.',
        'Valid Acc.'
    ])

    test_logger = Logger(
        os.path.join(state['save_dir'], 'test-log.txt'))


    test_logger.set_names([
        'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.',
        'Valid Acc.'
    ])

    
    ## run testset before training
    test_loss, test_acc = test(rank, testloader, model, criterion, -1)
    
    # Train and val
    for epoch in range(start_epoch, state['epochs']):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, state['epochs'],
                                             state['lr']))
        
        
        # BEFORE_RUN: accuracy_log
        accuracy_log = []
        train_loss, train_acc = train(rank, trainloader, model, criterion, optimizer, epoch, accuracy_log, scheduler, selector)
        test_loss, test_acc = test(rank, testloader, model, criterion, epoch)

        # append logger file
        train_logger.append(
            [state['lr'], train_loss, test_loss, train_acc, test_acc])
        
        
        test_logger.append(
            [state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            checkpoint=state['save_dir'])

    train_logger.close()
    test_logger.close()

    print('Best acc:')
    print(best_acc)

def train(rank, trainloader, model, criterion, optimizer, epoch, accuracy_log, scheduler, selector):
    global global_step
    
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    
    
    num_backprops = 0
    
    if selector is not None:
        selector.init_for_this_epoch(epoch)
    
    for batch_idx, (inputs, targets, index) in enumerate(trainloader):
        optimizer.zero_grad()
        model.train()
        
        
        if index.nelement() == 0:
            continue
        
        #######################################
        # measure data loading time
        data_time.update(time.time() - end)
        
        inputs, targets = send_data_to_device(inputs, rank), send_data_to_device(targets, rank)
        #######################################
        
        
        if SELECTIVE_BACKPROP:
            r"""
            Select inputs and targets
            """
            
            if epoch >= SB_WARMUP_EPOCH:
                inputs, targets, upweights = selector.update_examples(model, criterion, inputs, targets, index, losses, epoch)

            
            
            if inputs.nelement() == 0:
                bar.next()
                continue
        
        
        ## compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        
        if SELECTIVE_BACKPROP and UPWEIGHT_LOSS and (epoch >= SB_WARMUP_EPOCH):
            #print ("\n" + str(upweights.max().item()))
            loss = loss * upweights
        
        #####################################################################################################################
        # TODO:niel.hu (MERGE)
        top1and2 = F.softmax(outputs, dim=-1).topk(2)[0].detach()
        confidence = (top1and2[..., 0] - top1and2[..., 1])
        
        if LOG_TO_DISK:
            train_logger.blob['conf'] += [confidence.cpu().detach().numpy()]
        
         
        ######################
        
        #####################################################################################################################
        
        topk = (1, 5)
        pred, res = compute_pred(targets, outputs, topk)
        prec1, prec5 = res
        ##############################################################################
        
        if CORRECT_COMPACTION:
            r"""
            upweight losses
            """
            loss, index = selector.confcorrect_compaction(confidence, targets == pred[0], epoch, loss, index, rank)
         
        
        ################################################################################################################
        
        if LOG_TO_DISK:
            train_logger.blob['correct_pred'] += [index[targets == pred[0]].cpu().detach().numpy()]
            train_logger.blob['train_loss'] += [loss.mean().item()]
            train_logger.blob['train_pred1'] += [prec1.item() / 100]
        
        #######################################
        
        loss_mean = loss.mean()
        #######################################
        # measure accuracy and record loss
        losses.update(loss_mean.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        optimizer.zero_grad()
        
        
        if LOG_TO_DISK:
            train_logger.blob['backprops'] += [loss.nelement()]
            train_logger.blob['lr'] += [optimizer.param_groups[0]['lr']]
            
        num_backprops += loss.nelement()
        loss_mean.backward()
        optimizer.step()
        
        #######################################
        res = targets == pred[0]  # top-1
        
        # track iteration
        global_step += 1
        #######################################
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Global Step: {global_step} | Data: {data:.3f}s |  Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            global_step = global_step,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
        
        for _ in range(loss.nelement()):
            scheduler.step()
        
        
    ### SelectiveBP
    if LOG_TO_DISK:
        train_logger.blob['epoch_backprops'] += [num_backprops]  
        train_logger.blob['epoch_pred1'] += [top1.avg / 100]
    
    del inputs, targets, outputs, loss
    
    torch.cuda.empty_cache()
    bar.finish()
    return (losses.avg, top1.avg)


def test(rank, testloader, model, criterion, epoch):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    classes_accuracy = {i: AverageMeter() for i in range(num_classes)}

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(testloader):
            #######################################
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = send_data_to_device(inputs, rank), send_data_to_device(targets, rank)
            #######################################
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_mean = loss.mean()
            ###############################################################################
            # measure accuracy and record loss
            """Computes the precision@k for the specified values of k"""
            topk = (1, 5)
            pred, res = compute_pred(targets, outputs, topk)
            prec1, prec5 = res
            #################### update the result of this epoch
            res = targets == pred[0]  # top-1
            #######################################
            losses.update(loss_mean.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            
            ####################################### log accuracy
            
            if LOG_TO_DISK:
                test_logger.blob['pred1'] += [prec1.item() / 100]
            
            
            #######################################
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #######################################
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    
    if LOG_TO_DISK:
        test_logger.blob['epoch_pred1'] += [top1.avg / 100]
    
    del inputs, targets, outputs, loss
    torch.cuda.empty_cache()
    bar.finish()
    return (losses.avg, top1.avg)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    start = time.time()
    main()
    print (time.time() - start)