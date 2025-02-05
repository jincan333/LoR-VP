import functools
import os, shutil
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def try_cuda(*wargs):
    cuda_args = []
    for arg in wargs:
        if hasattr(arg, 'cuda'):
            cuda_args.append(arg.cuda(non_blocking=True))
        else:
            print(f"{arg} does not have a .cuda() method.")
            cuda_args.append(arg)
    return tuple(cuda_args)


def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def get_optimizer(parameters, args):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('optimizer should be one of [sgd, adam]')

    if args.scheduler == 'cosine':
        scheduler = WarmupCosineSchedule(optimizer, args.warmup_steps, args.total_train_steps)
    elif args.scheduler == 'linear':
        scheduler = WarmupLinearSchedule(optimizer, args.warmup_steps, args.total_train_steps)
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.total_train_steps * _) for _ in args.decreasing_step], gamma=0.1)
    else:
        raise ValueError('scheduler should be one of [cosine, multistep]')

    return optimizer, scheduler


def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))


def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))
