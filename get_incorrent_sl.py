import os
import re

# Folder containing influence score files
input_folder = 'influence_scores_seed5'
output_folder = 'filtered_influence_scores'
output_file = os.path.join(output_folder, 'final_filtered_influence_scores.txt')
initial_threshold = 50000  # Initial threshold value
threshold_increment = 10000  # Increment value for threshold

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Collect all results across files
all_results = []
unique_global_indices = {}

# Iterate through all .txt files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_file = os.path.join(input_folder, filename)

        # Read the file
        with open(input_file, 'r') as f:
            lines = f.readlines()

        ulb_indices = []
        salutary_labels = []
        influence_scores = []
        ground_truth = []
        pattern = re.compile(r"Index: (\d+), Salutary Label: (\d+), Ground Truth: (\d+), Influence Score: ([\d.]+)")

        # Parse the file content
        for line in lines:
            match = pattern.search(line)
            if match:
                index, salutary_label, gt, influence_score = match.groups()
                ulb_indices.append(int(index))
                salutary_labels.append(int(salutary_label))
                ground_truth.append(int(gt))
                influence_scores.append(float(influence_score))

        # Dynamically adjust the threshold and filter
        threshold = initial_threshold
        while True:
            filtered_indices = [ulb_indices[i] for i in range(len(ulb_indices)) if influence_scores[i] < threshold]
            filtered_salutary_labels = [salutary_labels[i] for i in range(len(ulb_indices)) if influence_scores[i] < threshold]
            filtered_ground_truth = [ground_truth[i] for i in range(len(ulb_indices)) if influence_scores[i] < threshold]
            filtered_influence_scores = [influence_scores[i] for i in range(len(ulb_indices)) if influence_scores[i] < threshold]
            if filtered_indices:
                break
            threshold += threshold_increment

        # Check for duplicate indices globally
        for i in range(len(filtered_indices)):
            index = filtered_indices[i]
            if index in unique_global_indices:
                # Update the salutary label for duplicate indices
                unique_global_indices[index]['salutary_label'] = filtered_salutary_labels[i]
            else:
                unique_global_indices[index] = {
                    'salutary_label': filtered_salutary_labels[i],
                    'ground_truth': filtered_ground_truth[i]
                }

# Prepare final results from unique global indices
all_results = [
    f"{index}, {unique_global_indices[index]['salutary_label']}, {unique_global_indices[index]['ground_truth']}\n"
    for index in unique_global_indices
]

# Write all results to the final file at once
with open(output_file, 'w') as f:
    f.writelines(all_results)

print(f"All filtered influence scores written to {output_file}")
