'''
miscellaneous functions: learning
'''
import os
import datetime
import gzip

import numpy as np

import torch
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


''' expand tensor '''
def expand_tensor(input, sample_size, do_unsqueeze):
    batch_size = input.size(0)
    if do_unsqueeze:
        sz_from = [-1]*(input.dim()+1)
        sz_from[1] = sample_size
        input_expanded = input.unsqueeze(1).expand(*sz_from).contiguous()

        sz_to = list(input.size())
        sz_to[0] = batch_size*sample_size
    else:
        assert input.size(1) == 1
        sz_from = [-1]*(input.dim())
        sz_from[1] = sample_size
        input_expanded = input.expand(*sz_from).contiguous()

        _sz_to = list(input.size())
        sz_to = _sz_to[0:1]+_sz_to[2:]
        sz_to[0] = batch_size*sample_size
    input_expanded_flattend = input_expanded.view(*sz_to)
    return input_expanded, input_expanded_flattend

''' cont out size '''
def conv_out_size(hin, kernel_size, stride=1, padding=0, dilation=1):
    hout = (hin + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
    return int(hout)

def deconv_out_size(hin, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    hout = (hin-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    return int(hout)


''' annealing '''
def annealing_func(val_init, val_fin, val_annealing, step):
    val = val_init + (val_fin  - val_init)  / float(val_annealing)  * float(min(val_annealing, step)) if val_annealing is not None else val_fin
    return float(val)


''' for monitoring lr '''
def get_lrs(optimizer):
    lrs = [float(param_group['lr']) for param_group in optimizer.param_groups]
    lr_max = max(lrs)
    lr_min = min(lrs)
    return lr_min, lr_max


''' save and load '''
def save_checkpoint(state, opt, filename='checkpoint.pth.tar', compress=False):
    filename = os.path.join(opt.path, filename)
    print("=> save checkpoint '{}'".format(filename))
    if compress:
        with gzip.open(filename, 'wb') as f:
            torch.save(state, f)
    else:
        torch.save(state, filename)
    print("=> done")

def load_checkpoint(opt, model=None, optimizer=None, filename='checkpoint.pth.tar', verbose=True, device=None, scheduler=None):
    filename = os.path.join(opt.path, filename)
    if os.path.isfile(filename):
        if verbose:
            print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=device) if device is not None else torch.load(filename)
        opt.start_episode  = checkpoint['start_episode']
        opt.offset_time    = checkpoint['offset_time']
        opt.total_numsteps = checkpoint['total_numsteps']
        opt.updates        = checkpoint['updates']
        opt.eval_steps     = checkpoint['eval_steps']
        opt.ckpt_steps     = checkpoint['ckpt_steps']
        if 'dataframe' in checkpoint:
            opt.dataframe = checkpoint['dataframe']
        else:
            opt.dataframe = None
        if model is not None:
            model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if verbose:
            print("=> loaded checkpoint '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

def load_replay_memory(memory, opt, filename='replay_memory.pth.tar', verbose=True, compress=True):
    filename = os.path.join(opt.path, filename)
    if os.path.isfile(filename):
        if verbose:
            print("=> loading replay memory '{}'".format(filename))
        if compress:
            try:
                with gzip.open(filename, 'rb') as f:
                    checkpoint = torch.load(f)
            except OSError:
                checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename)
        opt.start_episode  = checkpoint['start_episode']
        opt.offset_time    = checkpoint['offset_time']
        opt.total_numsteps = checkpoint['total_numsteps']
        opt.updates        = checkpoint['updates']
        opt.eval_steps     = checkpoint['eval_steps']
        opt.ckpt_steps     = checkpoint['ckpt_steps']
        return checkpoint['replay_memory']
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        return memory

''' log '''
def print_args(args):
    configuration_setup=''
    for arg in vars(args):
        configuration_setup+=' {} : {}'.format(str(arg),str(getattr(args, arg)))
        configuration_setup+='\n'
    return configuration_setup

def logging(s, path=None, filename='log.txt'):
    # print
    print(s)

    # save
    if path is not None:
        assert path, 'path is not define. path: {}'.format(path)
        with open(os.path.join(path, filename), 'a+') as f_log:
            f_log.write(s + '\n')

def get_time():
    return datetime.datetime.now().strftime('%y%m%d-%H:%M:%S')
    #return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')
