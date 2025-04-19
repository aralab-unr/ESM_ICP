import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import random

# Path to your folder
matrix_folder = "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res"
file_paths = glob.glob(os.path.join(matrix_folder, "corr_matrix_*.txt"))

# Natural sort by iteration number
def natural_sort_key(path):
    numbers = re.findall(r'\d+', os.path.basename(path))
    return int(numbers[0]) if numbers else -1

file_paths_sorted = sorted(file_paths, key=natural_sort_key)
n_files = len(file_paths_sorted)

# Pick 6 random middle indices
middle_indices = list(range(1, n_files - 1))
random_indices = sorted(random.sample(middle_indices, 6))

# Final list of 8 indices
selected_indices = [0] + random_indices + [n_files - 1]

# Load selected matrices
matrices = []
titles = []

for idx in selected_indices:
    path = file_paths_sorted[idx]
    try:
        mat = np.loadtxt(path)
        if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
            matrices.append(mat)
            titles.append(f"Iteration {idx}")
        else:
            print(f"⚠️ Skipping malformed matrix: {path}")
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")

# Plot 2 rows × 4 columns
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    sns.heatmap(matrices[i], cmap="cividis", square=True, cbar=False, ax=ax)
    ax.set_title(titles[i], fontsize=12)
    ax.axis('off')

plt.suptitle("Similarity Matrices $\\mathbf{M}$ Over Selected Iterations", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("similarity_matrix_sampled.png", dpi=300)
plt.show()
