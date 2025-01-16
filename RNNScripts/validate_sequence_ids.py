import numpy as np
import argparse
import os

def validate_sequence_ids(data_dir, sequence_ids_path, sequences_path):
    """
    Validate that each frame's sequence ID corresponds to its actual sequence directory.

    Args:
        data_dir (str): Path to the directory containing all sequences.
        sequence_ids_path (str): Path to the sequence_ids.npy file.
        sequences_path (str): Path to the sequences.npy file.
    """
    sequences = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    sequence_ids = np.load(sequence_ids_path)
    sequences_data = np.load(sequences_path, allow_pickle=True)

    current_index = 0
    total_frames_expected = sum(len(seq) for seq in sequences_data)

    for seq_id, seq in enumerate(sequences_data):
        num_frames = len(seq)
        if current_index + num_frames > len(sequence_ids):
            print(f"Error: Not enough sequence IDs. Expected at least {current_index + num_frames}, found {len(sequence_ids)}.")
            return False
        for i in range(num_frames):
            if sequence_ids[current_index] != seq_id:
                print(f"Mismatch at frame {current_index}: Expected sequence ID {seq_id}, Found {sequence_ids[current_index]}")
                return False
            current_index += 1

    if current_index != len(sequence_ids):
        print(f"Warning: sequence_ids.npy has more entries ({len(sequence_ids)}) than expected ({current_index}).")
    else:
        print("Validation successful: All sequence IDs are correctly mapped.")

    return True

def main():
    parser = argparse.ArgumentParser(description="Validate sequence_ids.npy mapping.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing all sequences.')
    parser.add_argument('--sequence_ids_path', type=str, required=True,
                        help='Path to the sequence_ids.npy file.')
    parser.add_argument('--sequences_path', type=str, required=True,
                        help='Path to the sequences.npy file.')
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        exit(1)
    if not os.path.exists(args.sequence_ids_path):
        print(f"Error: sequence_ids.npy file '{args.sequence_ids_path}' does not exist.")
        exit(1)
    if not os.path.exists(args.sequences_path):
        print(f"Error: sequences.npy file '{args.sequences_path}' does not exist.")
        exit(1)

    validate_sequence_ids(args.data_dir, args.sequence_ids_path, args.sequences_path)

if __name__ == "__main__":
    main()
