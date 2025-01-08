import numpy as np
import argparse
from sklearn.cluster import KMeans
import joblib
import os

def load_reduced_features(reduced_features_path):
    """
    Load the reduced feature embeddings from a .npy file.
    """
    print(f"Loading reduced features from {reduced_features_path}...")
    if not os.path.exists(reduced_features_path):
        raise FileNotFoundError(f"The file {reduced_features_path} does not exist.")
    reduced_features = np.load(reduced_features_path)
    print(f"Reduced features shape: {reduced_features.shape}")
    return reduced_features

def perform_kmeans(reduced_features, n_clusters=7):
    """
    Apply K-Means clustering to the reduced features.
    """
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_features)
    print("K-Means clustering completed.")
    return cluster_labels, kmeans

def save_cluster_labels(labels, output_path):
    """
    Save cluster labels to a .npy file.
    """
    print(f"Saving cluster labels to {output_path}...")
    np.save(output_path, labels)
    print("Cluster labels saved successfully.")

def save_kmeans_model(kmeans, output_path):
    """
    Save the K-Means model for future use.
    """
    print(f"Saving K-Means model to {output_path}...")
    joblib.dump(kmeans, output_path)
    print("K-Means model saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Clustering with K-Means")
    parser.add_argument('--reduced_features', type=str, default='features_reduced.npy',
                        help='Path to the .npy file containing reduced feature embeddings.')
    parser.add_argument('--output_labels', type=str, default='cluster_labels.npy',
                        help='Path to save the cluster labels.')
    parser.add_argument('--kmeans_model', type=str, default='kmeans_model.joblib',
                        help='Path to save the K-Means model.')
    parser.add_argument('--n_clusters', type=int, default=7,
                        help='Number of clusters for K-Means.')
    args = parser.parse_args()

    # Load reduced features
    reduced_features = load_reduced_features(args.reduced_features)

    # Perform K-Means clustering
    cluster_labels, kmeans = perform_kmeans(reduced_features, n_clusters=args.n_clusters)

    # Save cluster labels
    save_cluster_labels(cluster_labels, args.output_labels)

    # Save K-Means model
    save_kmeans_model(kmeans, args.kmeans_model)

if __name__ == '__main__':
    main()

