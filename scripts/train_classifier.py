import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os

def load_features_and_labels(features_path, labels_path, mapping_path):
    """
    Load feature embeddings and corresponding labels.
    """
    print(f"Loading features from {features_path}...")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"The file {features_path} does not exist.")
    features = np.load(features_path)
    
    print(f"Loading cluster labels from {labels_path}...")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"The file {labels_path} does not exist.")
    cluster_labels = np.load(labels_path)
    
    print(f"Loading cluster mapping from {mapping_path}...")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"The file {mapping_path} does not exist.")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        cluster_mapping = json.load(f)
    
    # Map cluster IDs to emotion labels
    emotion_labels = [cluster_mapping[str(label)] for label in cluster_labels]
    
    return features, emotion_labels

def encode_labels(emotion_labels):
    """
    Encode string labels to integers.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    encoded_labels = le.fit_transform(emotion_labels)
    return encoded_labels, le

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    """
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    print("Training completed.")
    return clf

def evaluate_classifier(clf, X_test, y_test, label_encoder):
    """
    Evaluate the classifier and print metrics.
    """
    print("Evaluating classifier...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def save_classifier(clf, label_encoder, output_clf, output_le):
    """
    Save the trained classifier and label encoder.
    """
    print(f"Saving classifier to {output_clf}...")
    joblib.dump(clf, output_clf)
    print(f"Saving label encoder to {output_le}...")
    joblib.dump(label_encoder, output_le)
    print("Classifier and label encoder saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Train a Classifier using Cluster Labels")
    parser.add_argument('--features_path', type=str, default='features_reduced.npy',
                        help='Path to the reduced feature embeddings.')
    parser.add_argument('--labels_path', type=str, default='cluster_labels.npy',
                        help='Path to the cluster labels.')
    parser.add_argument('--mapping_path', type=str, default='cluster_mapping.json',
                        help='Path to the cluster-to-emotion mapping JSON.')
    parser.add_argument('--output_clf', type=str, default='emotion_classifier.joblib',
                        help='Path to save the trained classifier.')
    parser.add_argument('--output_le', type=str, default='label_encoder.joblib',
                        help='Path to save the label encoder.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split.')
    args = parser.parse_args()

    # Load features and labels
    features, emotion_labels = load_features_and_labels(
        args.features_path, args.labels_path, args.mapping_path
    )

    # Encode labels
    encoded_labels, label_encoder = encode_labels(emotion_labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=args.test_size, random_state=42, stratify=encoded_labels
    )

    # Train classifier
    clf = train_logistic_regression(X_train, y_train)

    # Evaluate classifier
    evaluate_classifier(clf, X_test, y_test, label_encoder)

    # Save classifier and label encoder
    save_classifier(clf, label_encoder, args.output_clf, args.output_le)

if __name__ == '__main__':
    main()
