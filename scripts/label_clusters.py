import numpy as np
import argparse
import json
import os

def load_cluster_labels(labels_path):
    """
    Load cluster labels from a .npy file.
    """
    print(f"Loading cluster labels from {labels_path}...")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"The file {labels_path} does not exist.")
    labels = np.load(labels_path)
    print(f"Cluster labels shape: {labels.shape}")
    return labels

def create_cluster_label_mapping(num_clusters):
    """
    Manually assign emotion labels to cluster IDs based on visual inspection.
    """
    mapping = {}
    print(f"\nPlease assign an emotion label to each of the {num_clusters} clusters.")
    for cluster_id in range(num_clusters):
        label = input(f"Enter label for Cluster {cluster_id}: ")
        mapping[str(cluster_id)] = label.strip()
    return mapping

def save_mapping(mapping, output_path):
    """
    Save the cluster-to-emotion mapping to a JSON file.
    """
    print(f"\nSaving cluster mapping to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)
    print("Cluster mapping saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Map Clusters to Emotion Labels")
    parser.add_argument('--labels_path', type=str, default='cluster_labels.npy',
                        help='Path to the .npy file containing cluster labels.')
    parser.add_argument('--num_clusters', type=int, default=7,
                        help='Number of clusters.')
    parser.add_argument('--output_mapping', type=str, default='cluster_mapping.json',
                        help='Path to save the cluster-to-emotion mapping.')
    args = parser.parse_args()

    # Load cluster labels
    labels = load_cluster_labels(args.labels_path)

    # Create mapping
    mapping = create_cluster_label_mapping(args.num_clusters)

    # Save mapping
    save_mapping(mapping, args.output_mapping)

if __name__ == '__main__':
    main()
