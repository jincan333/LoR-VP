import logging
import traceback
import torch
import torch.distributed as dist
import sys
import os
import time
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import argparse
import json
from functools import partial
from timm.utils import accuracy, AverageMeter
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_and_data import get_torch_dataset, get_model, get_specified_model
from utils.exp_utils import set_seed, get_optimizer, try_cuda, AverageMeter, reduce_tensor
from utils.label_mapping import generate_label_mapping_by_frequency, label_mapping_base, CustomNetwork
from visual_prompt import LoR_VP


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification')
    # visual_prompt
    parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'mobilenet', 'vit_b_16', 'vit_b_16_1k', 'vit_l_16', 'resnet50_21k', 'vit_b_16_21k', 'swin_b', 'test', 'vit_b_32_1k'])
    parser.add_argument('--pretrain_path', type=str, default='../model/ViT-B_16.npz', help='pretrained model directory')
    parser.add_argument('--head_init', type=str, default='pretrain', choices=['uniform', 'normal', 'xavier_uniform', 'kaiming_normal', 'zero', 'default', 'pretain'])
    parser.add_argument('--randomcrop', type=int, default=0, choices=[1, 0])
    parser.add_argument("--shuffle", default=1, type=int, help="whether shuffle the train dataset")
    parser.add_argument("--is_observe", default=0, type=int, help="whether observe images, vp, and features")
    # ILM-VP 
    #   cifar10, cifar100, gtsrb, svhn, abide 32
    #   food101, eurosat, sun397, ucf101, stanfordcars, flowers102, dtd, oxfordpets  varies
    parser.add_argument('--input_size', type=int, default=224, help='image size before prompt')
    parser.add_argument('--output_size', type=int, default=224, help='image size before prompt')
    parser.add_argument('--downstream_mapping', type=str, default='lp', choices=['origin', 'fm', 'ilm', 'flm', 'lp'])
    parser.add_argument('--mapping_freq', type=int, default=1, help='frequency of label mapping')
    parser.add_argument('--prompt_method', type=str, default='lor_vp', choices=['lor_vp'])
    parser.add_argument('--bar_width', type=int, default=4)
    parser.add_argument('--bar_height', type=int, default=224)
    parser.add_argument('--init_method', type=str, default='zero,normal')
    # all
    parser.add_argument('--dataset', type=str, default='cifar100', choices = ['cifar10', 'cifar100', 'tiny_imagenet', 'imagenet'])
    parser.add_argument('--datadir', type=str, default='../data', help='data directory')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0,help='random seed')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'linear'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--eval_frequency', type=int, default=5)
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument('--fp16', default=True, type=bool, help="Whether to use 16-bit float precision instead of 32-bit")
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--global_step', type=int, default=0)
    parser.add_argument('--save_path', type=str,  default='ckpt/LoR-VP'+randomhash, help='path to save the final model')
    parser.add_argument('--exp_name', type=str, default='LoR-VP')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--specified_path', type=str, default='')
    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    return args


def evaluate(network, loader, args):
    network.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()
    for batch in loader:
        with torch.no_grad():
            inputs, targets = try_cuda(*batch[:2])
            with torch.cuda.amp.autocast():
                pred = network(inputs)
            loss = F.cross_entropy(pred, targets)
            acc = accuracy(pred, targets, topk=(1,))[0]
            if args.local_rank != -1:
                acc = reduce_tensor(acc)
                loss = reduce_tensor(loss)
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc.item(), inputs.size(0))
    end = time.time()
    eval_elapsed = end - start
    log_str = '| Eval {:3d} at epoch {:>8d} | time: {:5.2f}s | acc: {:5.2f} | loss: {:5.2f} |'.format(
            args.epoch // args.eval_frequency, args.epoch, eval_elapsed, acc_meter.avg, loss_meter.avg)
    logger.info(log_str)
    if args.local_rank in [-1, 0]:
        wandb.log({'test_acc': acc_meter.avg, 'test_loss': loss_meter.avg}, step=args.global_step)

    return acc_meter.avg


def train(network, optimizer, scheduler, loader, args):
    scaler = torch.cuda.amp.GradScaler()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(loader):
        inputs, targets = try_cuda(inputs, targets)
        with torch.cuda.amp.autocast():
            pred = network(inputs)
        loss = F.cross_entropy(pred, targets) / args.gradient_accumulation_steps
        acc = accuracy(pred, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), inputs.size(0))
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (idx+1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scaler.unscale_(optimizer)
            if args.gradient_accumulation_steps > 1:
                torch.nn.utils.clip_grad_norm_(network.module.get_tunable_params(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            args.global_step += 1
            if args.local_rank in [-1, 0]:
                wandb.log({'lr': scheduler.get_last_lr()[0], 'train_loss': loss_meter.avg, 'train_acc': acc_meter.avg}, step=args.global_step)

    end = time.time()
    train_elapsed = end - start
    log_str = '| epoch {:3d} | time: {:5.2f} | lr: {:.3e} | acc: {:5.2f} | loss: {:5.2f} |'.format(
            args.epoch, train_elapsed, scheduler.get_last_lr()[0], acc_meter.avg, loss_meter.avg)
    logger.info(log_str)
    

def apply_label_mapping(network, visual_prompt, train_loader, args):
    if args.epoch == 0:
        network = CustomNetwork(network, visual_prompt, None, None, args).cuda()
    if args.downstream_mapping in ['ilm', 'flm']:
        mapping_sequence = generate_label_mapping_by_frequency(network, train_loader)
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        logger.info(f'mapping sequence: {mapping_sequence.tolist()}')
    elif args.downstream_mapping == 'origin':
        mapping_sequence = torch.arange(args.class_cnt)
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        logger.info(f'mapping sequence: {mapping_sequence.tolist()}')
    else:
        label_mapping = None
        mapping_sequence = None

    network.label_mapping = label_mapping
    network.mapping_sequence = mapping_sequence

    return network


def main():
    try:
        start_time = time.time()
        args = parse_args()
        # setup logger
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger.info(json.dumps(vars(args), indent=4))
        # Set device and distributed training
        if args.local_rank == -1:
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(int(args.gpu))
            args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            args.n_gpu = 1
            logger.info(f"Process {args.local_rank} is using GPU {torch.cuda.current_device()}")
            logger.info(f"Initialized process group; rank: {dist.get_rank()}, world size: {dist.get_world_size()}")
        args.device = device
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
        set_seed(args.seed)

        # dataset
        train_loader, test_loader = get_torch_dataset(args, 'vp')

        # init network
        logger.info(f"network: {args.network}")
        network=get_model(args.network, args)
        network.to(device)
        total_params = sum(p.numel() for p in network.parameters())
        logger.info(f"Total number of network parameters: {total_params:,}")
        if args.local_rank in [-1, 0]:
            os.makedirs(args.save_path, exist_ok=True)
            wandb_username=os.environ.get('WANDB_USER_NAME')
            wandb_key=os.environ.get('WANDB_API_KEY')    
            wandb.login(key=wandb_key)
            wandb.init(project='LoR_VP_'+args.dataset, entity=wandb_username, name=args.exp_name)

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        # visual prompt
        if args.prompt_method == 'LoR_VP':
            visual_prompt = LoR_VP(args, normalize=args.normalize).to(device)
        else:
            visual_prompt = None

        # label mapping
        network = apply_label_mapping(network, visual_prompt, train_loader, args)
        tunable_params = network.get_tunable_params()
        tunable_params_num = sum(p.numel() for p in tunable_params)
        args.warmup_steps = int(args.total_train_steps * args.warmup_ratio)
        optimizer, scheduler = get_optimizer(tunable_params, args)
        logger.info(f"Tunable parameters: {tunable_params_num}")
        logger.info(f"network: {network}")
        if args.specified_path:
            network = get_specified_model(network, args)

        if args.local_rank != -1:
            network = DDP(network, device_ids=[args.local_rank], output_device=args.local_rank)

        logger.info(f"{'*'*20} Train and Evaluate Visual Prompt {'*'*20}")
        logger.info("  Total optimization steps = %d", args.total_train_steps)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Eval batch size = %d", args.eval_batch_size)

        acc = evaluate(network, test_loader, args)
        best_acc = acc
        for epoch in range(1, args.epochs + 1):
            args.epoch = epoch
            train(network, optimizer, scheduler, train_loader, args)
            if args.downstream_mapping == 'ilm' and epoch % args.mapping_freq == 0:
                network =  apply_label_mapping(network, visual_prompt, train_loader, args)
            if epoch % args.eval_frequency == 0:
                acc = evaluate(network, test_loader, args)
            if args.local_rank in [-1, 0]:
                if acc > best_acc:
                    best_acc = acc                 

        logger.info(f"ckpt path: {args.save_path}")
        if args.local_rank in [-1, 0]:
            wandb.finish()

        logger.info("Final Accuracy: \t%f" % best_acc)
        logger.info(f"{'*'*20} Finish Training {'*'*20}")
        end_time = time.time()
        total_time_seconds = end_time - start_time
        total_time_hours = total_time_seconds / 3600
        logger.info(f"Total time taken: {total_time_seconds} seconds")
        logger.info(f"Total time taken: {total_time_hours:.2f} hours")

    except Exception:
        logger.error(traceback.format_exc())
        if args.local_rank != -1:
            dist.destroy_process_group()
        return float('NaN')


if __name__ == '__main__':
    main()
