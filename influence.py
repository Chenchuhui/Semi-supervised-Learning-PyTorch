from typing import Any

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.arguments import FactorArguments

import argparse
import logging
import math
import os
import random
import shutil
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset.cv_dataset import DATASET_GETTERS
from dataset.cifar import CIFAR10SSL, CIFAR100SSL
from dataset.utils import CBSBatchSampler
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.utils.data import TensorDataset

class SSLTask(Task):
    def compute_train_loss(self, batch: Any, model: nn.Module, sample: bool = False) -> torch.Tensor:
        """Compute loss for supervised training."""
        # batch = to_tensor(batch)  # Ensure batch elements are tensors
        x_w, x_s, target = batch
        logits_w = model(x_w)
        logits_s = model(x_s)
        Lx = F.cross_entropy(logits_w, target, reduction='mean') + F.cross_entropy(logits_s, target, reduction='mean')
        return Lx

    def compute_measurement(self, batch: Any, model: nn.Module) -> torch.Tensor:
        inputs_v, targets_v, _ = batch
        logits_v = model(inputs_v)
        Lx = F.cross_entropy(logits_v, targets_v, reduction='mean')
        return Lx
    

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, group=2):
    s = list(x.shape)
    size = x.shape[0] // group
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, group=2):
    s = list(x.shape)
    size = x.shape[0] // group
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'stl10'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--train-iteration', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=10, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--CBS', action='store_true',
                        help='use Circulum Batch Size. Faster Convergence')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--alpha', default=0.7, type=float,
                        help='CBS index')
    parser.add_argument('--val_split', default=0.1, type=float,
                        help='validation split')
    parser.add_argument('--img-size', type=int, default=32,
                        help='Image Size')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                            help='Crop Ratio')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--preaug', action='store_true', 
                    help='If set, use preprocessed augmented data to speed up training and better utilize GPU resources.')
    parser.add_argument('--rep', type=int, default=1, 
                    help='Repetition Index. Number of repetitive augmentations of the same image (optional, valid only in preaug mode).')

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            print(args.num_classes)
            model = models.WideResNet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropRate=0,
                                            num_classes=args.num_classes)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    args.factor_args = FactorArguments(
        strategy="ekfac",  # Choose from "identity", "diagonal", "kfac", or "ekfac".
        use_empirical_fisher=False,
        amp_dtype=None,
        amp_scale=2.0**16,
        has_shared_parameters=False,

        # Settings for covariance matrix fitting.
        covariance_max_examples=100_000,
        covariance_data_partitions=1,
        covariance_module_partitions=1,
        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        
        # Settings for Eigendecomposition.
        eigendecomposition_dtype=torch.float64,
        
        # Settings for Lambda matrix fitting.
        lambda_max_examples=100_000,
        lambda_data_partitions=1,
        lambda_module_partitions=1,
        use_iterative_lambda_aggregation=False,
        offload_activations_to_cpu=False,
        per_sample_gradient_dtype=torch.float32,
        lambda_dtype=torch.float32,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset in ['cifar10', 'svhn', 'stl10']:
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')
    
    # Expand unlabeled dataset: each unlabeled sample is repeated for each labele
    unlabeled_images = [(img_w, img_s) for (img_w, img_s), _, _ in unlabeled_dataset]
    # Separate lists for img_w and img_s
    expanded_img_w = []
    expanded_img_s = []
    expanded_labels = []

    for (img_w, img_s) in unlabeled_images:
        for label in range(args.num_classes):
            expanded_img_w.append(img_w)
            expanded_img_s.append(img_s)
            expanded_labels.append(label)
    
    # Convert lists to tensors
    expanded_img_w = torch.stack(expanded_img_w)
    expanded_img_s = torch.stack(expanded_img_s)
    expanded_labels = torch.tensor(expanded_labels)

    # Create new unlabeled dataset using TensorDataset
    unlabeled_dataset = TensorDataset(expanded_img_w, expanded_img_s, expanded_labels)

    # print unlabeled dataset size
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    # unlabeled_sampler = CBSBatchSampler(unlabeled_expanded_dataset, args.batch_size*args.mu, args.alpha, args.train_iteration, args.total_steps)
    labeled_trainloader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # if args.CBS:
    #     unlabeled_sampler = CBSBatchSampler(unlabeled_expanded_dataset, args.batch_size*args.mu, args.alpha, args.train_iteration, args.total_steps)
    #     unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_sampler=unlabeled_sampler, num_workers=args.num_workers)
    # else:
    #     unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=args.batch_size*args.mu, shuffle=True, num_workers=args.num_workers, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.train_iteration)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        ema_model = None

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        log = Logger(os.path.join(args.out, 'log.txt'), resume=True)
    else:
        log = Logger(os.path.join(args.out, 'log.txt'))
        log.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Test Loss', 'Test Acc.', 'Epoch Time'])

    # Ensure the output directory exists
    os.makedirs(args.out, exist_ok=True)
    
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    # Model warm up: train the model with labeled data for n epochs
    print("start epch", args.start_epoch)
    if args.start_epoch < args.warmup:
        train(args, labeled_trainloader, unlabeled_dataset, val_dataset, test_loader,
            model, optimizer, ema_model, scheduler, log, warmup=True)
        args.start_epoch = args.warmup
    
    # Train the model with labeled and unlabeled data
    train(args, labeled_trainloader, unlabeled_dataset, val_dataset, test_loader,
          model, optimizer, ema_model, scheduler, log, warmup=False)
    
def updated_ulb_dataset(unlabeled_dataset, val_dataset, model, args, epoch):
    model = prepare_model(model=model, task=SSLTask())
    analyzer = Analyzer(analysis_name="cifar10-ekfac-val", model=model, task=SSLTask())
    analyzer.fit_all_factors(factors_name=f"SSL_factor_{epoch}", dataset=unlabeled_dataset, factor_args=args.factor_args)
    analyzer.compute_pairwise_scores(
        scores_name=f"SSL_score_{epoch}",
        factors_name=f"SSL_factor_{epoch}",
        query_dataset=val_dataset,
        train_dataset=unlabeled_dataset,
        per_device_query_batch_size=1024,
    )

    scores = analyzer.load_pairwise_scores(scores_name=f"SSL_score_{epoch}")

    # Ensure 'all_modules' exists in scores
    if 'all_modules' not in scores:
        raise KeyError("Key 'all_modules' not found in scores.")

    # Sum each row to get the influence score of each sample
    influence_scores = scores['all_modules'].sum(dim=0)
    positive_indices = (influence_scores > 0).nonzero(as_tuple=True)[0]  # Get positive indices

    # Check if the filtered dataset is empty
    if len(positive_indices) == 0:
        raise ValueError("No samples with positive influence scores found.")

    # Filter out the samples with negative influence scores
    unlabeled_dataset = torch.utils.data.Subset(unlabeled_dataset, positive_indices.tolist())
    return unlabeled_dataset


def train(args, labeled_trainloader, unlabeled_dataset, val_dataset, test_loader,
          model, optimizer, ema_model, scheduler, log, warmup: bool):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    # unlabeled_iter = iter(unlabeled_trainloader)

    epoch_time = AverageMeter()
    if warmup:
        epochs = args.warmup
    else:
        epochs = args.epochs
    model.train()
    # if ulb_sampler is not None:
    #     ulb_sampler.reset_epoch(args.start_epoch)
    for epoch in range(args.start_epoch, epochs):
        epoch_start = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        
        if not warmup:
            # Deep copy of the model for influence computation
            model_copy = copy.deepcopy(model) 
            # Update the unlabeled dataset filtering out the samples with negative influence scores
            unlabeled_dataset = updated_ulb_dataset(unlabeled_dataset, val_dataset, model_copy, args, epoch)
            unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=args.batch_size*args.mu, shuffle=True, num_workers=args.num_workers, drop_last=True)
            unlabeled_iter = iter(unlabeled_trainloader)

        # if ulb_sampler is not None:
        #     ulb_sampler.reset_epoch(epoch)
        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x, _ = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, _ = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            if not warmup:
                try:
                    inputs_u_w, inputs_u_s, targets_u = next(unlabeled_iter)
                    # error occurs ↓
                    # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
                except:
                    if args.world_size > 1:
                        print("Enter set epoch")
                        unlabeled_epoch += 1
                        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            # ulb_batch_size = inputs_u_w.shape[0]
            if warmup:
                inputs_x = interleave(inputs_x, 2).to(args.device)
                targets_x = targets_x.to(args.device)
                logits_x = de_interleave(model(inputs_x), 2)
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                loss = Lx
            else:
                inputs_x = inputs_x.to(args.device)
                inputs_u_w, inputs_u_s, targets_u = inputs_u_w.to(args.device), inputs_u_s.to(args.device), targets_u.to(args.device)
                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2).to(args.device)
                targets_x = targets_x.to(args.device)
                logits = model(inputs)
                logits = de_interleave(logits, 2)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                ulb_batch_size = inputs_u_w.shape[0]

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                Lu_w = F.cross_entropy(logits_u_w, targets_u,
                                    reduction='mean')
                Lu_s = F.cross_entropy(logits_u_s, targets_u,
                                    reduction='mean')
                Lu = Lu_w + Lu_s

                loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            if not warmup:
                losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.train_iteration,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,))
                p_bar.update()

        epoch_time.update(time.time()-epoch_start)
        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            log.append([losses.avg, losses_x.avg, losses_u.avg, test_loss, test_acc, epoch_time.avg])

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
