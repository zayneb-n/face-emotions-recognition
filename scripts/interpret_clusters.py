import numpy as np
import argparse
import random
import cv2
import matplotlib.pyplot as plt
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

def load_image_paths(paths_txt):
    """
    Load image paths from a text file.
    Each line should contain: <index>\t<image_path>
    """
    print(f"Loading image paths from {paths_txt}...")
    if not os.path.exists(paths_txt):
        raise FileNotFoundError(f"The file {paths_txt} does not exist.")
    image_paths = []
    with open(paths_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                _, path = parts
                image_paths.append(path)
            else:
                print(f"Skipping invalid line: {line.strip()}")
    print(f"Total image paths loaded: {len(image_paths)}")
    return image_paths

def show_cluster_examples(cluster_id, labels, image_paths, num_examples=5):
    """
    Display random examples from a specific cluster.
    """
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        print(f"No samples found in cluster {cluster_id}.")
        return
    
    selected_indices = random.sample(list(indices), min(num_examples, len(indices)))
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(selected_indices):
        img_path = image_paths[idx]
        if not os.path.exists(img_path): 
            print(f"Could not find image at {img_path}.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image at {img_path}.")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, num_examples, i+1)
        plt.imshow(img_rgb)
        plt.axis('off')
    plt.suptitle(f"Cluster {cluster_id} Examples")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Interpret Clusters by Visualizing Examples")
    parser.add_argument('--labels_path', type=str, default='cluster_labels.npy',
                        help='Path to the .npy file containing cluster labels.')
    parser.add_argument('--paths_txt', type=str, default='paths.txt',
                        help='Path to the text file containing image paths.')
    parser.add_argument('--num_clusters', type=int, default=7,
                        help='Number of clusters.')
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of examples to display per cluster.')
    args = parser.parse_args()

    # Load cluster labels and image paths
    labels = load_cluster_labels(args.labels_path)
    image_paths = load_image_paths(args.paths_txt)

    # Display examples for each cluster
    for cluster_id in range(args.num_clusters):
        print(f"\nDisplaying examples for Cluster {cluster_id}...")
        show_cluster_examples(cluster_id, labels, image_paths, num_examples=args.num_examples)

if __name__ == '__main__':
    main()
