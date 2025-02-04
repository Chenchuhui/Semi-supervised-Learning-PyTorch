import torch
from torch.utils.data import Sampler


def CBS(u, alpha, t, T):
    x = 1 - t / T
    return max(1, int(u * (1 - x / ((1 - alpha) + alpha * x))))

class CBSBatchSampler(Sampler):
    def __init__(self, data_source, max_batch_size=64*7, alpha=0.7, train_iteration=1024, total_iters=1024*1024):
        self.data_source = data_source
        self.max_batch_size = max_batch_size  # This is the max batch size
        self.alpha = alpha
        self.train_iteration = train_iteration
        self.total_iters = total_iters
        self.epoch = 0

    def __iter__(self):
        indices = torch.randperm(len(self.data_source)).tolist()  # Shuffle dataset indices
        current_iter = 1

        while current_iter <= self.train_iteration:
            # Compute batch size using CBS function per iteration
            batch_size = CBS(self.max_batch_size, self.alpha, current_iter + self.epoch * self.train_iteration, self.total_iters)
            batch_size = min(batch_size, len(indices))  # Ensure it does not exceed dataset size

            # Sample batch
            batch_indices = indices[:batch_size]
            indices = indices[batch_size:]  # Remove used indices
            
            # If we run out of data, reshuffle and restart
            if len(indices) < batch_size:
                indices = torch.randperm(len(self.data_source)).tolist()

            yield batch_indices
            current_iter += 1  # Update iteration count

    def reset_epoch(self, epoch):
        self.epoch = epoch