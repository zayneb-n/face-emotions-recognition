# scripts/generate_sequence_ids.py

import os
import argparse
import numpy as np

def generate_sequence_ids(data_dir, output_path, valid_extensions=None):
    """
    Generates a sequence_ids.npy file by assigning a unique ID to each sequence.
    Skips any sequence directories that do not contain any valid frames.

    Args:
        data_dir (str): Path to the directory containing all sequences.
        output_path (str): Path to save the generated sequence_ids.npy file.
        valid_extensions (list, optional): List of valid image file extensions. Defaults to ['.jpg', '.jpeg', '.png'].
    """
    if valid_extensions is None:
        valid_extensions = ['.jpg', '.jpeg', '.png']

    sequence_ids = []
    current_id = 0

    # List all subdirectories in the data_dir
    sequences = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(sequences)} potential sequences in '{data_dir}'.")

    skipped_sequences = 0
    processed_sequences = 0

    for seq in sequences:
        seq_path = os.path.join(data_dir, seq)
        # List all files in the sequence directory with valid extensions
        frames = sorted([
            f for f in os.listdir(seq_path)
            if os.path.isfile(os.path.join(seq_path, f)) and os.path.splitext(f)[1].lower() in valid_extensions
        ])
        num_frames = len(frames)

        if num_frames == 0:
            print(f"Sequence '{seq}' is empty or contains no valid frames. Skipping.")
            skipped_sequences += 1
            continue

        print(f"Sequence '{seq}' has {num_frames} frames.")
        sequence_ids.extend([current_id] * num_frames)
        current_id += 1
        processed_sequences += 1

    print(f"\nTotal sequences processed: {processed_sequences}")
    print(f"Total sequences skipped: {skipped_sequences}")
    print(f"Total frames mapped: {len(sequence_ids)}")

    # Convert to NumPy array
    sequence_ids_array = np.array(sequence_ids)
    
    # Save to .npy file
    np.save(output_path, sequence_ids_array)
    print(f"sequence_ids.npy saved to '{output_path}'.")
    

def main():
    parser = argparse.ArgumentParser(description="Generate sequence_ids.npy by mapping each frame to its sequence.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing all sequences (e.g., dataset/AFEW-VA-processed/002/).')
    parser.add_argument('--output_path', type=str, default='models/sequence_ids.npy',
                        help='Path to save the generated sequence_ids.npy file.')
    parser.add_argument('--valid_extensions', type=str, nargs='*', default=['.jpg', '.jpeg', '.png'],
                        help='List of valid image file extensions (e.g., .jpg .png).')
    args = parser.parse_args()

    # Check if data_dir exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        exit(1)

    generate_sequence_ids(args.data_dir, args.output_path, args.valid_extensions)

if __name__ == "__main__":
    main()
