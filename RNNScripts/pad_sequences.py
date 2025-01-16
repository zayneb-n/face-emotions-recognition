# scripts/pad_sequences.py

import numpy as np
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
from collections import Counter

def load_sequences(sequences_path):
    """
    Load sequences from a .npy file.
    """
    print(f"Loading sequences from {sequences_path}...")
    sequences = np.load(sequences_path, allow_pickle=True)
    print(f"Total sequences loaded: {len(sequences)}")
    return sequences

def load_encoded_labels(labels_path):
    """
    Load encoded labels from a .npy file.
    """
    print(f"Loading encoded labels from {labels_path}...")
    encoded_labels = np.load(labels_path)
    print(f"Total labels loaded: {len(encoded_labels)}")
    return encoded_labels

def pad_sequences_data(sequences, max_length, padding='post', truncating='post', value=0.0):
    """
    Pad sequences to the same length.
    """
    print(f"Padding sequences to a maximum length of {max_length}...")
    padded = pad_sequences(sequences, maxlen=max_length, dtype='float32',
                          padding=padding, truncating=truncating, value=value)
    print(f"Padded sequences shape: {padded.shape}")
    return padded

def save_padded_data(padded_sequences, labels, output_sequences_path, output_labels_path):
    """
    Save padded sequences and labels to .npy files.
    """
    print(f"Saving padded sequences to {output_sequences_path}...")
    np.save(output_sequences_path, padded_sequences)
    
    print(f"Saving labels to {output_labels_path}...")
    np.save(output_labels_path, labels)
    
    print("Padded sequences and labels saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Pad Sequences for GRU Model")
    parser.add_argument('--sequences_path', type=str, default='models/sequences.npy',
                        help='Path to the sequences of features.')
    parser.add_argument('--labels_path', type=str, default='models/sequence_labels.npy',
                        help='Path to the sequence labels.')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum sequence length. If not set, uses the 95th percentile of sequence lengths.')
    parser.add_argument('--padding', type=str, default='post',
                        choices=['pre', 'post'],
                        help='Padding type.')
    parser.add_argument('--truncating', type=str, default='post',
                        choices=['pre', 'post'],
                        help='Truncating type.')
    parser.add_argument('--padding_value', type=float, default=0.0,
                        help='Value to use for padding.')
    parser.add_argument('--output_sequences', type=str, default='models/padded_sequences.npy',
                        help='Path to save the padded sequences.')
    parser.add_argument('--output_labels', type=str, default='models/padded_sequence_labels.npy',
                        help='Path to save the sequence labels.')
    parser.add_argument('--sliding_window', action='store_true',
                        help='Enable sliding window approach for long sequences.')
    parser.add_argument('--window_size', type=int, default=30,
                        help='Window size for sliding window.')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for sliding window.')
    args = parser.parse_args()
    
    # Load data
    sequences = load_sequences(args.sequences_path)
    encoded_labels = load_encoded_labels(args.labels_path)
    
    # Determine maximum sequence length
    if args.max_length is None:
        # Calculate 95th percentile
        lengths = [len(seq) for seq in sequences]
        max_length = int(np.percentile(lengths, 95))
        print(f"Determined maximum sequence length based on 95th percentile: {max_length}")
    else:
        max_length = args.max_length
        print(f"Using specified maximum sequence length: {max_length}")
    
    # Optional: Implement sliding window to handle very long sequences
    if args.sliding_window:
        print("Applying sliding window approach to handle long sequences...")
        WINDOW_SIZE = args.window_size
        STEP_SIZE = args.step_size
        new_sequences = []
        new_labels = []
        
        for seq, label in zip(sequences, encoded_labels):
            if len(seq) > WINDOW_SIZE:
                for i in range(0, len(seq) - WINDOW_SIZE + 1, STEP_SIZE):
                    window = seq[i:i + WINDOW_SIZE]
                    window_label = label  # Assuming label is consistent; adjust if necessary
                    new_sequences.append(window)
                    new_labels.append(window_label)
            else:
                new_sequences.append(seq)
                new_labels.append(label)
        
        sequences = new_sequences
        encoded_labels = new_labels
        print(f"Total sequences after sliding window: {len(sequences)}")
    
    # Pad sequences
    padded_sequences = pad_sequences_data(sequences, max_length=args.window_size if args.sliding_window else max_length,
                                         padding=args.padding, truncating=args.truncating, value=args.padding_value)
    
    # Save padded sequences and labels
    save_padded_data(padded_sequences, encoded_labels, args.output_sequences, args.output_labels)

if __name__ == '__main__':
    main()
