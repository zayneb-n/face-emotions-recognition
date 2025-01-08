import numpy as np
import argparse
from sklearn.decomposition import PCA

def load_features(features_path):
    """
    Load the feature embeddings from a .npy file.
    """
    print(f"Loading features from {features_path}...")
    features = np.load(features_path)
    print(f"Features shape: {features.shape}")
    return features

def apply_pca(features, n_components=50):
    """
    Apply PCA to reduce feature dimensions.
    """
    print(f"Applying PCA to reduce dimensions to {n_components}...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(features)
    print(f"Reduced features shape: {reduced_features.shape}")
    return reduced_features, pca

def save_reduced_features(reduced_features, output_path):
    """
    Save the reduced feature embeddings to a .npy file.
    """
    print(f"Saving reduced features to {output_path}...")
    np.save(output_path, reduced_features)
    print("Reduced features saved successfully.")

def save_pca_model(pca, output_path):
    """
    Save the PCA model for future use.
    """
    import joblib
    print(f"Saving PCA model to {output_path}...")
    joblib.dump(pca, output_path)
    print("PCA model saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Dimensionality Reduction with PCA")
    parser.add_argument('--features_path', type=str, default='features.npy',
                        help='Path to the .npy file containing feature embeddings.')
    parser.add_argument('--output_reduced', type=str, default='features_reduced.npy',
                        help='Path to save the reduced feature embeddings.')
    parser.add_argument('--pca_model', type=str, default='pca_model.joblib',
                        help='Path to save the PCA model.')
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of principal components for PCA.')
    args = parser.parse_args()

    # Load features
    features = load_features(args.features_path)

    # Apply PCA
    reduced_features, pca = apply_pca(features, n_components=args.n_components)

    # Save reduced features
    save_reduced_features(reduced_features, args.output_reduced)

    # Save PCA model
    save_pca_model(pca, args.pca_model)

if __name__ == '__main__':
    main()
