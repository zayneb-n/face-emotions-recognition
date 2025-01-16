# scripts/prepare_sequences.py

import numpy as np
import argparse
from collections import Counter
import os
import sys

def load_data(features_path, labels_path, sequences_path):
    """
    Load feature embeddings, cluster labels, and sequence identifiers.
    """
    print(f"Loading features from {features_path}...")
    features = np.load(features_path)  # Shape: (num_frames, feature_dim)
    
    print(f"Loading cluster labels from {labels_path}...")
    labels = np.load(labels_path)      # Shape: (num_frames,)
    
    print(f"Loading sequence IDs from {sequences_path}...")
    sequence_ids = np.load(sequences_path)  # Shape: (num_frames,)
    
    return features, labels, sequence_ids

def create_sequences(features, labels, sequence_ids):
    """
    Organize frames into sequences based on sequence IDs.
    """
    sequences = []
    sequence_labels = []
    
    current_seq = []
    current_labels = []
    current_id = sequence_ids[0]
    
    for i in range(len(sequence_ids)):
        seq_id = sequence_ids[i]
        if seq_id == current_id:
            current_seq.append(features[i])
            current_labels.append(labels[i])
        else:
            sequences.append(current_seq)
            # Assign the most common label in the sequence
            most_common_label = Counter(current_labels).most_common(1)[0][0]
            sequence_labels.append(most_common_label)
            # Reset for the next sequence
            current_seq = [features[i]]
            current_labels = [labels[i]]
            current_id = seq_id
    
    # Append the last sequence
    if current_seq:
        sequences.append(current_seq)
        most_common_label = Counter(current_labels).most_common(1)[0][0]
        sequence_labels.append(most_common_label)
    
    print(f"Total sequences created: {len(sequences)}")
    return sequences, sequence_labels

def save_sequences(sequences, sequence_labels, output_features_path, output_labels_path):
    """
    Save sequences and their labels to .npy files.
    """
    print(f"Saving sequences to {output_features_path}...")
    np.save(output_features_path, sequences, allow_pickle=True)
    
    print(f"Saving sequence labels to {output_labels_path}...")
    np.save(output_labels_path, sequence_labels)
    
    print("Sequences and labels saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Prepare Sequential Data for GRU")
    parser.add_argument('--features_path', type=str, default='models/features_reduced.npy',
                        help='Path to the PCA-reduced feature embeddings.')
    parser.add_argument('--labels_path', type=str, default='models/cluster_labels.npy',
                        help='Path to the cluster labels.')
    parser.add_argument('--sequences_path', type=str, default='models/sequence_ids.npy',
                        help='Path to the sequence identifiers.')
    parser.add_argument('--output_features', type=str, default='models/sequences.npy',
                        help='Path to save the sequences of features.')
    parser.add_argument('--output_labels', type=str, default='models/sequence_labels.npy',
                        help='Path to save the sequence labels.')
    args = parser.parse_args()
    
    # Check if sequence_ids.npy exists; if not, create it based on video/frame grouping
    if not os.path.exists(args.sequences_path):
        print(f"Sequence IDs file {args.sequences_path} not found. Please create it by grouping frames into sequences.")
        # Implement logic to generate sequence_ids.npy if needed
        # This may involve parsing video filenames or directory structures
        sys.exit(1)
    
    # Load data
    features, labels, sequence_ids = load_data(args.features_path, args.labels_path, args.sequences_path)
    
    # Create sequences
    sequences, sequence_labels = create_sequences(features, labels, sequence_ids)
    
    # Save sequences and labels
    save_sequences(sequences, sequence_labels, args.output_features, args.output_labels)

if __name__ == '__main__':
    main()
