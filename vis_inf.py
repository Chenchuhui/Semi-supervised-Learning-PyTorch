from safetensors.torch import load_file

# Load the safetensors file
pairwise_scores = load_file("influence_results/cifar10-ekfac-val/scores_SSL_score_10/pairwise_scores.safetensors")

# Print keys (tensor names)
print("Stored Tensors:", pairwise_scores.keys())

# Access a specific tensor
layer1_scores = pairwise_scores["all_modules"]
print("Layer 1 Scores Shape:", layer1_scores.shape)
# print first column and save this in a new txt file
with open("layer1_scores.txt", "w") as f:
    for i in range(layer1_scores.shape[0]):
        f.write(str(layer1_scores[i, 0].item()) + "\n")

# influence_scores = layer1_scores.sum(dim=0)
# positive_indices = influence_scores > 0
# unlabeled_dataset = influence_scores[positive_indices]

# print(influence_scores)
# print(positive_indices)
# print(unlabeled_dataset)