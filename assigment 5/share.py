# ============================================================
# LiDAR Processing Assignment 5
# Full working solution divided into:
#   Task 1 - Ground level detection
#   Task 2 - DBSCAN eps optimization + clustering
#   Task 3 - Find largest cluster (catenary)
#
# This code works for both dataset1.npy and dataset2.npy
# and saves all required plots automatically.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


# ============================================================
# General settings
# ============================================================

DATASETS = ["dataset1.npy", "dataset2.npy"]
OUTPUT_FOLDER = "images"
MIN_SAMPLES = 5   # required by DBSCAN and also used in k-distance plot

# Create folder for saved plots if it does not already exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def show_cloud_3d(points, title="Point Cloud", sample_step=1):
    """
    Show a 3D scatter plot of the point cloud.
    sample_step=1 means show all points.
    sample_step=10 means show every 10th point, useful if plotting is slow.
    """
    pts = points[::sample_step]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def save_histogram(z_values, dataset_name, bins=120):
    """
    Save histogram of Z values for Task 1.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(z_values, bins=bins)
    plt.title(f"{dataset_name} - Z Value Histogram")
    plt.xlabel("Z (height)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, f"{dataset_name}_histogram.png")
    plt.savefig(output_path, dpi=200)
    plt.show()
    plt.close()

    return output_path


def get_ground_level(pcd, bins=120):
    """
    Task 1:
    Estimate ground level using histogram of Z values.
    The idea:
      - take all Z values
      - build a histogram
      - find the bin with the highest frequency
      - the center of that bin is the estimated ground level
    """
    z_values = pcd[:, 2]

    hist, bin_edges = np.histogram(z_values, bins=bins)
    max_bin_index = np.argmax(hist)

    # Take center of the most frequent bin
    ground_level = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    return ground_level


def remove_ground_points(pcd, ground_level):
    """
    Keep only points above the estimated ground level.
    """
    return pcd[pcd[:, 2] > ground_level]


def compute_k_distance(points, min_samples=5):
    """
    Task 2:
    Compute k-distance values for elbow plot.
    For DBSCAN, a common choice is to use the distance to the k-th nearest neighbor,
    where k = min_samples.
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(points)

    distances, _ = neighbors_fit.kneighbors(points)

    # Take the distance to the k-th nearest neighbor
    k_distances = distances[:, min_samples - 1]

    # Sort for elbow plot
    k_distances = np.sort(k_distances)

    return k_distances


def estimate_eps_from_k_distance(k_distances, percentile=98):
    """
    Automatically estimate eps from the k-distance curve.

    Why this works:
    The elbow is often near the upper part of the sorted distances.
    A simple and stable automatic choice is a high percentile.

    """
    eps = np.percentile(k_distances, percentile)
    return eps


def save_elbow_plot(k_distances, eps, dataset_name):
    """
    Save elbow plot for Task 2.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.axhline(y=eps, linestyle='--', label=f"Estimated eps = {eps:.3f}")
    plt.title(f"{dataset_name} - Elbow Plot for DBSCAN")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{MIN_SAMPLES}-NN distance")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, f"{dataset_name}_elbow_plot.png")
    plt.savefig(output_path, dpi=200)
    plt.show()
    plt.close()

    return output_path


def run_dbscan(points, eps, min_samples=5):
    """
    Run DBSCAN and return labels.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(points)
    return labels


def save_cluster_plot(points, labels, dataset_name):
    """
    Save 2D cluster plot using X and Y.
    Noise points will also appear based on their label.
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab20", s=2)
    plt.title(f"{dataset_name} - DBSCAN Clusters ({n_clusters} clusters)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, f"{dataset_name}_clusters.png")
    plt.savefig(output_path, dpi=200)
    plt.show()
    plt.close()

    return output_path, n_clusters


def find_largest_cluster_by_xy_span(points, labels):
    """
    Task 3:
    Find the cluster with the largest span in X and Y.
    Ignore noise label = -1.

    Span is calculated as:
        (max_x - min_x) + (max_y - min_y)

    The cluster with the largest total XY span is treated as the catenary.
    """
    unique_labels = set(labels)

    best_label = None
    best_points = None
    best_span = -1

    for label in unique_labels:
        if label == -1:
            # Ignore noise
            continue

        cluster_points = points[labels == label]

        if len(cluster_points) == 0:
            continue

        x_span = cluster_points[:, 0].max() - cluster_points[:, 0].min()
        y_span = cluster_points[:, 1].max() - cluster_points[:, 1].min()
        total_span = x_span + y_span

        if total_span > best_span:
            best_span = total_span
            best_label = label
            best_points = cluster_points

    return best_label, best_points, best_span


def save_catenary_plot(catenary_points, dataset_name):
    """
    Save plot of the detected catenary cluster.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(catenary_points[:, 0], catenary_points[:, 1], s=2)
    plt.title(f"{dataset_name} - Catenary Cluster")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, f"{dataset_name}_catenary.png")
    plt.savefig(output_path, dpi=200)
    plt.show()
    plt.close()

    return output_path


# ============================================================
# Main processing loop for both datasets
# ============================================================

all_results = []

for dataset_file in DATASETS:
    print("\n" + "=" * 70)
    print(f"Processing {dataset_file}")
    print("=" * 70)

    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]

    # --------------------------------------------------------
    # Load point cloud
    # --------------------------------------------------------
    pcd = np.load(dataset_file)
    print(f"Loaded {dataset_file} with shape: {pcd.shape}")

    # ========================================================
    # TASK 1 - Find ground level
    # ========================================================
    print("\n--- TASK 1: Ground level detection ---")

    ground_level = get_ground_level(pcd, bins=120)
    print(f"Estimated ground level: {ground_level:.6f}")

    hist_path = save_histogram(pcd[:, 2], dataset_name, bins=120)
    print(f"Histogram saved to: {hist_path}")

    pcd_above_ground = remove_ground_points(pcd, ground_level)
    print(f"Points above ground: {pcd_above_ground.shape}")

    # ========================================================
    # TASK 2 - Find optimized eps and cluster
    # ========================================================
    print("\n--- TASK 2: DBSCAN eps optimization ---")

    k_distances = compute_k_distance(pcd_above_ground, min_samples=MIN_SAMPLES)

    # Automatic estimate of eps
    # It can change percentile to 97, 98, or 99 if testing nedded.
    eps = estimate_eps_from_k_distance(k_distances, percentile=98)
    print(f"Estimated eps: {eps:.6f}")

    elbow_path = save_elbow_plot(k_distances, eps, dataset_name)
    print(f"Elbow plot saved to: {elbow_path}")

    labels = run_dbscan(pcd_above_ground, eps=eps, min_samples=MIN_SAMPLES)

    cluster_plot_path, n_clusters = save_cluster_plot(pcd_above_ground, labels, dataset_name)
    print(f"Cluster plot saved to: {cluster_plot_path}")
    print(f"Number of clusters found (excluding noise): {n_clusters}")

    noise_points = np.sum(labels == -1)
    print(f"Noise points: {noise_points}")

    # ========================================================
    # TASK 3 - Find largest cluster (catenary)
    # ========================================================
    print("\n--- TASK 3: Find largest cluster (catenary) ---")

    best_label, catenary_points, best_span = find_largest_cluster_by_xy_span(pcd_above_ground, labels)

    if catenary_points is None:
        print("No valid cluster found.")
        continue

    min_x = catenary_points[:, 0].min()
    min_y = catenary_points[:, 1].min()
    max_x = catenary_points[:, 0].max()
    max_y = catenary_points[:, 1].max()

    print(f"Catenary cluster label: {best_label}")
    print(f"XY span score: {best_span:.6f}")
    print(f"min(x) = {min_x:.6f}")
    print(f"min(y) = {min_y:.6f}")
    print(f"max(x) = {max_x:.6f}")
    print(f"max(y) = {max_y:.6f}")

    catenary_plot_path = save_catenary_plot(catenary_points, dataset_name)
    print(f"Catenary plot saved to: {catenary_plot_path}")

    # Store results for summary
    all_results.append({
        "dataset": dataset_name,
        "ground_level": ground_level,
        "eps": eps,
        "clusters": n_clusters,
        "noise_points": noise_points,
        "catenary_label": best_label,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "histogram_plot": hist_path,
        "elbow_plot": elbow_path,
        "cluster_plot": cluster_plot_path,
        "catenary_plot": catenary_plot_path,
    })


# ============================================================
# Final printed summary
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for result in all_results:
    print(f"\nDataset: {result['dataset']}")
    print(f"  Ground level : {result['ground_level']:.6f}")
    print(f"  Optimal eps  : {result['eps']:.6f}")
    print(f"  Clusters     : {result['clusters']}")
    print(f"  Noise points : {result['noise_points']}")
    print(f"  Catenary min(x): {result['min_x']:.6f}")
    print(f"  Catenary min(y): {result['min_y']:.6f}")
    print(f"  Catenary max(x): {result['max_x']:.6f}")
    print(f"  Catenary max(y): {result['max_y']:.6f}")

print("\nDone. All plots are saved inside the 'images' folder.")