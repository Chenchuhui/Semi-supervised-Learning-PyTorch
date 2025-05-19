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
from dataset.utils import CBSBatchSampler, CBS
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.utils.data import TensorDataset

class SSLTask(Task):
    def compute_train_loss(self, batch: Any, model: nn.Module, sample: bool = False) -> torch.Tensor:
        """Compute loss for supervised training."""
        # batch = to_tensor(batch)  # Ensure batch elements are tensors
        x_w, target, _ = batch
        logits_w = model(x_w)
        Lx = F.cross_entropy(logits_w, target, reduction='sum')
        return Lx
    
    def compute_measurement(self, batch, model):
        inputs, labels = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

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

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

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
    parser.add_argument('--warmup', default=10, type=int,
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
    
    # Set unlabeled dataset target to -1
    unlabeled_dataset.targets = [-1] * len(unlabeled_dataset)
    
    # print unlabeled dataset size
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    # unlabeled_sampler = CBSBatchSampler(unlabeled_expanded_dataset, args.batch_size*args.mu, args.alpha, args.train_iteration, args.total_steps)
    labeled_trainloader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

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

    unlabeled_labeled_indices = []

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
        
        # If there is a file unlabeled_labeled_indices.txt, load the indices
        if os.path.exists(f"{args.out}/unlabeled_labeled_indices.txt"):
            with open(f"{args.out}/unlabeled_labeled_indices.txt", "r") as f:
                labels = []
                for line in f:
                    index, label, gt = line.strip().split(", ")
                    unlabeled_labeled_indices.append(int(index))
                    labels.append(int(label))
                unlabeled_labeled_indices = list(set(unlabeled_labeled_indices))
                print(f"Unlabeled labeled indices: {unlabeled_labeled_indices}")
                # Update the targets of the unlabeled dataset
                if isinstance(unlabeled_dataset.targets, list):
                    unlabeled_dataset.targets = torch.tensor(unlabeled_dataset.targets)
                unlabeled_dataset.targets[unlabeled_labeled_indices] = torch.tensor(labels, dtype=torch.long)
    else:
        log = Logger(os.path.join(args.out, 'log.txt'))
        log.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Val Loss', 'Val Acc.', 'Test Loss', 'Test Acc.', 'Epoch Time'])

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
            model, optimizer, ema_model, scheduler, log, warmup=True, unlabeled_labeled_indices=unlabeled_labeled_indices)
        args.start_epoch = args.warmup
    
    # Train the model with labeled and unlabeled data
    train(args, labeled_trainloader, unlabeled_dataset, val_dataset, test_loader,
          model, optimizer, ema_model, scheduler, log, warmup=False, unlabeled_labeled_indices=unlabeled_labeled_indices)

import matplotlib.pyplot as plt

def visualize(scores, ds_val, ds_train, descending=True):
    val_indices = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]
    
    for i in val_indices:
        fig, axs = plt.subplots(ncols=7, figsize=(15, 3))
        fig.suptitle("Top Influential Training Images")

        # Get query image and label
        query_img, query_label = ds_val[i]
        query_img = query_img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

        axs[0].imshow(query_img)
        axs[0].axis("off")
        axs[0].set_title(f"Query\nLabel: {query_label}")

        axs[1].axis("off")  # Spacer column

        # Get top 5 influential training sample indices
        top_idxs = scores['all_modules'][i].argsort(descending=descending)[:5]

        for ii, idx in enumerate(top_idxs):
            train_img, train_label, gt = ds_train[idx]
            train_img = train_img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

            axs[ii + 2].imshow(train_img)
            axs[ii + 2].axis("off")
            axs[ii + 2].set_title(f"Label: {train_label}, GT: {gt}")

        plt.tight_layout()
        plt.show()

def visualize_train_to_val(scores, ds_val, ds_train, descending=True):
    train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    
    for i in train_indices:
        fig, axs = plt.subplots(ncols=7, figsize=(15, 3))
        fig.suptitle("Top Influential Training Images")

        # Get query image and label
        train_img,_, train_label, gt = ds_train[i]
        train_img = train_img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

        axs[0].imshow(train_img)
        axs[0].axis("off")
        axs[0].set_title(f"Query\nLabel: {train_label}, GT: {gt}")

        axs[1].axis("off")  # Spacer column

        # Get top 5 influential validation sample indices
        influence_matrix = scores['all_modules']
        influence_matrix_t = influence_matrix.transpose(0, 1)
        top_idxs = influence_matrix_t[i].argsort(descending=descending)[:5]

        for ii, idx in enumerate(top_idxs):
            val_img, val_label = ds_val[idx]
            val_img = val_img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

            axs[ii + 2].imshow(val_img)
            axs[ii + 2].axis("off")
            axs[ii + 2].set_title(f"Label: {val_label},")

        plt.tight_layout()
        plt.show()

def updated_ulb_dataset(unlabeled_dataset, val_dataset, model, args, epoch, unlabeled_indices):
    # Delete all files in the folder
    if os.path.exists(f"./influence_results/cifar10-ekfac-influence-verify-{args.seed}"):
        shutil.rmtree(f"./influence_results/cifar10-ekfac-influence-verify-{args.seed}")
    model = prepare_model(model=model, task=SSLTask())
    analyzer = Analyzer(analysis_name=f"cifar10-ekfac-influence-verify-{args.seed}", model=model, task=SSLTask())
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
    assert len(influence_scores) == len(unlabeled_dataset), "Influence scores length mismatch."
    N = len(influence_scores)
    group_size = args.num_classes
    # Number of top indices to select per group
    top_k = 1

    # Sanity check
    assert N % group_size == 0, "Total samples should be divisible by group size."

    top_indices_per_group = []

    for i in range(0, N, group_size):
        group = influence_scores[i:i + group_size]
        top = torch.topk(group, k=top_k)
        # Store the original indices (not just within-group)
        top_indices_per_group.extend((top.indices + i).tolist())

    # if epoch < args.warmup*3:
    #     filtered_indices = top_indices_per_group
    # else:


    # influence_threshold = 50000
    
    # while True:
    #     filtered_indices = [idx for idx in top_indices_per_group if influence_scores[idx] < influence_threshold]
    #     if filtered_indices:
    #         break
    #     influence_threshold += 10000

    # print(f"Influence threshold: {influence_threshold}")
        # increase the threshold
    # positive_indices = (influence_scores > 0).nonzero(as_tuple=True)[0]  # Get positive indices

    # After getting the score, create a txt file store current index, salutary label, ground truth label, and influence score
    
    with open(f"./influence_scores_seed{args.seed}/influence_scores_{epoch}.txt", "w") as f:
        correct_label = 0
        for idx in top_indices_per_group:
            _, label, gt = unlabeled_dataset[idx]
            if label == gt:
                correct_label += 1
            f.write(f"Index: {unlabeled_indices[idx // args.num_classes]}, Salutary Label: {label}, Ground Truth: {gt}, Influence Score: {influence_scores[idx]}\n")
        f.write(f"Correct labels: {correct_label}/{len(top_indices_per_group)}\n")
    
    # visualize(scores, val_dataset, unlabeled_dataset, descending=True)

    return torch.tensor(top_indices_per_group)

def train(args, labeled_trainloader, unlabeled_dataset, val_dataset, test_loader,
          model, optimizer, ema_model, scheduler, log, warmup: bool, unlabeled_labeled_indices):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)

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
            # Calculate number of unlabeled data size gonna use
            total = 0
            if args.CBS:
                for t in range(epoch * args.train_iteration, (epoch + 1) * args.train_iteration):
                    total += CBS(args.batch_size, args.alpha, t, args.total_steps)
                print(f"Total unlabeled data size: {total}")
                total = len(unlabeled_dataset) if total > len(unlabeled_dataset) else total
            else:
                total = 512
            # Create a subset of the unlabeled dataset randomly
            indices = torch.randperm(len(unlabeled_dataset))[:total]
            ulb_dataset = unlabeled_dataset.select_subset(indices)

            # Expand the dataset while keeping track of the original indices
            expanded_img_w = []
            expanded_labels = []
            ground_truths = []

            for (img_w, _), _, gt, _ in ulb_dataset:
                for label in range(args.num_classes):
                    expanded_img_w.append(img_w)
                    expanded_labels.append(label)
                    ground_truths.append(gt)

            expanded_img_w = torch.stack(expanded_img_w)
            expanded_labels = torch.tensor(expanded_labels)
            ground_truths = torch.tensor(ground_truths)

            # Create the expanded dataset with original indices
            expanded_dataset = TensorDataset(expanded_img_w, expanded_labels, ground_truths)

            # Deep copy of the model for influence computation
            model_copy = copy.deepcopy(model)

            # Get the indices with positive influence
            positive_indices = updated_ulb_dataset(expanded_dataset, val_dataset, model_copy, args, epoch, indices)
            
            selected_indices = indices[positive_indices // args.num_classes]

            # if len(positive_indices) == 2048:
            #     # Warmup phase: use the top-k indices
            #     ulb_dataset = unlabeled_dataset.select_subset(selected_indices)
            #     ulb_dataset.targets = expanded_labels[positive_indices]
            # else:
            unlabeled_labeled_indices.extend(selected_indices.tolist())

            # Remove duplicate element in the list
            unlabeled_labeled_indices = list(set(unlabeled_labeled_indices))
            
            # Update the targets of the unlabeled dataset
            if isinstance(unlabeled_dataset.targets, list):
                unlabeled_dataset.targets = torch.tensor(unlabeled_dataset.targets)

            unlabeled_dataset.targets[selected_indices] = expanded_labels[positive_indices]
            
            # Save the ulb_dataset to prevent unexpected interruption, save unlabeled_labeled_indices and labels
            # Save indices and labels
            with open(f"{args.out}/unlabeled_labeled_indices.txt", "w") as f:
                for idx in unlabeled_labeled_indices:
                    f.write(f"{idx}, {unlabeled_dataset.targets[idx]}, {unlabeled_dataset.gt[idx]}\n")
                
                # Calculate the accuracy and write to the file
                correct = (unlabeled_dataset.targets[unlabeled_labeled_indices] == unlabeled_dataset.gt[unlabeled_labeled_indices]).sum().item()
                f.write(f"Correct labels: {correct}/{len(unlabeled_labeled_indices)}\n")

            # Get the dataset with the targets not -1
            ulb_dataset = unlabeled_dataset.select_subset(unlabeled_labeled_indices)

            print(f"Unlabeled dataset size: {len(ulb_dataset)}")
            # unlabeled_ratio = len(ulb_dataset) / len(unlabeled_dataset)

            expand_factor = args.batch_size * args.mu * args.train_iteration // len(ulb_dataset)
            ulb_dataset.expand_data(expand_factor)

            ulb_dataloader = DataLoader(ulb_dataset, batch_size=args.batch_size*args.mu, shuffle=True, num_workers=args.num_workers, drop_last=True)
            unlabeled_iter = iter(ulb_dataloader)
            
        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x,_, _ = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, _, _ = next(labeled_iter)
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            if not warmup:
                try:
                    (inputs_u_w, inputs_u_s), targets_u, gt, _ = next(unlabeled_iter)
                    # print("Target:", targets_u)
                    # print("Ground Truth:", gt)
                    # # calculate the accuracy of the psudo label by comparing with gt
                    # print("Accuracy:", (targets_u == gt).sum().item() / targets_u.shape[0])

                    # targets_u = torch.tensor([index_to_target[idx.item()] for idx in indices]).to(args.device)
                except StopIteration:
                    unlabeled_iter = iter(ulb_dataloader)
                    (inputs_u_w, inputs_u_s), targets_u, gt, _ = next(unlabeled_iter)
                    # targets_u = torch.tensor([index_to_target[idx.item()] for idx in indices]).to(args.device)
            
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

                # ulb_batch_size = inputs_u_w.shape[0]

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                Lu_w = F.cross_entropy(logits_u_w, targets_u,
                                    reduction='mean')
                Lu_s = F.cross_entropy(logits_u_s, targets_u,
                                    reduction='mean')
                Lu = Lu_w + Lu_s

                loss = Lx + args.lambda_u * linear_rampup(epoch, args.epochs) * Lu

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
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            val_loss, val_acc = test(args, val_loader, test_model, epoch)
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.val_acc', val_acc, epoch)
            args.writer.add_scalar('train/5.val_loss', val_loss, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            log.append([losses.avg, losses_x.avg, losses_u.avg, val_loss, val_acc, test_loss, test_acc, epoch_time.avg])

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
