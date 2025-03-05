import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from kronfluence.task import Task
from typing import Any, Dict, List, Optional, Union
from kronfluence.arguments import FactorArguments

from kronfluence.analyzer import Analyzer, prepare_model
from dataset.cv_dataset import DATASET_GETTERS
import argparse
import logging

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
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--wdecay', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--CBS', action='store_true', default=True,
                    help='use Circulum Batch Size. Faster Convergence')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')
parser.add_argument('--use-ema', action='store_true', default=True,
                    help='use EMA model')
parser.add_argument('--ema-decay', default=0.999, type=float,
                    help='EMA decay rate')
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

device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

# Define the model and load the trained model weights.
def create_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        print(args.num_classes)
        model = models.WideResNet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropRate=0,
                                        num_classes=args.num_classes)
    model = model.to(device)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    return model

def to_tensor(batch):
    """Ensures that all batch elements are Tensors and moved to the correct device."""
    return tuple(t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device) for t in batch)

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
labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

unlabeled_images = [(img_w, img_s) for (img_w, img_s), _, _ in unlabeled_dataset]

# Separate lists for img_w and img_s
expanded_img_w = []
expanded_img_s = []
expanded_labels = []

for (img_w, img_s) in unlabeled_images:
    for label in range(10):
        expanded_img_w.append(img_w)  # Weak augmentation
        expanded_img_s.append(img_s)  # Strong augmentation
        expanded_labels.append(label)

# Convert lists to tensors
expanded_img_w_tensor = torch.stack(expanded_img_w)  # Shape: (N, C, H, W)
expanded_img_s_tensor = torch.stack(expanded_img_s)  # Shape: (N, C, H, W)
expanded_labels_tensor = torch.tensor(expanded_labels)  # Shape: (N,)

# Store as a dataset with multiple inputs
unlabeled_expanded_dataset = torch.utils.data.TensorDataset(
    expanded_img_w_tensor, expanded_img_s_tensor, expanded_labels_tensor
)

print(f"Expanded dataset size: {len(unlabeled_expanded_dataset)}")  # Should be 10x original size

# Define the task. See the Technical Documentation page for details.
task = SSLTask()
model = create_model(args)

# Prepare the model for influence computation.
model = prepare_model(model=model, task=task)
analyzer = Analyzer(analysis_name="cifar10-ekfac-val", model=model, task=task)
factor_args = FactorArguments(
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
# Fit all EKFAC factors for the given model.
analyzer.fit_all_factors(factors_name="my_factors", dataset=unlabeled_expanded_dataset, factor_args=factor_args)

# Compute all pairwise influence scores with the computed factors.
analyzer.compute_pairwise_scores(
    scores_name="my_scores",
    factors_name="my_factors",
    query_dataset=val_dataset,
    train_dataset=unlabeled_expanded_dataset,
    per_device_query_batch_size=1024,
)

# Load the scores with dimension `len(eval_dataset) x len(train_dataset)`.
scores = analyzer.load_pairwise_scores(scores_name="my_scores")

print(scores)