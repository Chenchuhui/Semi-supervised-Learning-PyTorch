from safetensors.torch import load_file
import torch

# Load the safetensors file
pairwise_scores = load_file("influence_results/cifar10-ekfac-val/scores_SSL_score_10/pairwise_scores.safetensors")

# Print keys (tensor names)
print("Stored Tensors:", pairwise_scores.keys())

# Access a specific tensor
scores = pairwise_scores["all_modules"]

influence_scores = scores['all_modules'].sum(dim=0)

N = len(influence_scores)
group_size = 10
top_k = 3

# Sanity check
assert N % group_size == 0, "Total samples should be divisible by group size."

top2_indices_per_group = []

for i in range(0, N, group_size):
    group = influence_scores[i:i + group_size]
    top2 = torch.topk(group, k=top_k)
    # Store the original indices (not just within-group)
    top2_indices_per_group.extend((top2.indices + i).tolist())

print("Top 2 indices from each group of 10:", top2_indices_per_group)
positive_indices = (influence_scores > 0).nonzero(as_tuple=True)[0]  # Get positive indices

# Check if the filtered dataset is empty
if len(top2_indices_per_group) == 0:
    raise ValueError("No samples with positive influence scores found.")

# After getting the score, create a txt file store current index, salutary label, ground truth label, and influence score

with open(f"./influence_scores_seed1/influence_scores_top3.txt", "w") as f:
    correct_label = 0
    for idx in top2_indices_per_group:
        img_w, img_s, label, gt = unlabeled_dataset[idx]
        if label == gt:
            correct_label += 1
        f.write(f"Index: {idx}, Salutary Label: {label}, Ground Truth: {gt}, Influence Score: {influence_scores[idx]}\n")
    f.write(f"Correct labels: {correct_label}/{len(top2_indices_per_group)}\n")