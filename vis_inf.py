from safetensors.torch import load_file

# Load the safetensors file
pairwise_scores = load_file("influence_results/cifar10-ekfac/scores_my_scores/pairwise_scores.safetensors")

# Print keys (tensor names)
print("Stored Tensors:", pairwise_scores.keys())

# Access a specific tensor
layer1_scores = pairwise_scores["all_modules"]
print("Layer 1 Scores Shape:", layer1_scores.shape)
print(layer1_scores[0][3999])
