import cv2
import os
import argparse

def detect_and_preprocess_face(img_path, face_cascade, output_size=(224, 224)):
    """
    Detect exactly ONE face in the image using the LBP cascade:
      - Convert to gray
      - Detect faces (if any)
      - Pick the largest bounding box
      - Crop & resize to output_size
    Returns the cropped face (BGR) or None if no face is found.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None  # Could not load image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        return None  # No face detected

    # Pick the LARGEST face by area
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_roi = img[y:y+h, x:x+w]

    # Resize
    face_resized = cv2.resize(face_roi, output_size)
    return face_resized

def preprocess_sequences(input_dir, output_dir,
                         cascade_path='lbpcascade_frontalface_improved.xml',
                         output_size=(224, 224), keep_placeholder=False):
    """
    Processes each subfolder (sequence) in input_dir:
      - For each frame, detects only ONE face (the largest).
      - Crops and resizes to output_size.
      - Saves the face as the same filename in output_dir.
      - If no face is found and keep_placeholder=True, saves a black image placeholder.
        Otherwise, it skips that frame.
    """
    # Load LBP Cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"Cannot load cascade file at {cascade_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Traverse each sequence directory
    sequence_dirs = [d for d in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, d))]

    for seq_name in sequence_dirs:
        seq_path = os.path.join(input_dir, seq_name)
        out_seq_path = os.path.join(output_dir, seq_name)
        os.makedirs(out_seq_path, exist_ok=True)

        # Process each frame (assuming .png files)
        frame_files = sorted([f for f in os.listdir(seq_path) if f.lower().endswith('.png')])
        for frame_file in frame_files:
            frame_path = os.path.join(seq_path, frame_file)
            processed_face = detect_and_preprocess_face(
                frame_path, face_cascade, output_size=output_size
            )

            if processed_face is not None:
                # Save the largest-face crop
                save_path = os.path.join(out_seq_path, frame_file)
                cv2.imwrite(save_path, processed_face)
            else:
                if keep_placeholder:
                    # Create a black placeholder
                    blank_img = cv2.imread(frame_path)
                    if blank_img is None:
                        continue  # in case frame_path is invalid
                    blank_img = cv2.resize(blank_img, output_size)
                    blank_gray = cv2.cvtColor(blank_img, cv2.COLOR_BGR2GRAY)
                    blank_bgr = cv2.cvtColor(blank_gray, cv2.COLOR_GRAY2BGR)
                    save_path = os.path.join(out_seq_path, frame_file)
                    cv2.imwrite(save_path, blank_bgr)
                else:
                    print(f"[WARNING] No face found in {frame_path}, skipping.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='Dataset/AFEW-VA',
                        help='Path to the folder with raw sequences.')
    parser.add_argument('--output_dir', type=str, default='Dataset/AFEW-VA-processed',
                        help='Path to save cropped faces.')
    parser.add_argument('--cascade', type=str, default='scripts/lbpcascade_frontalface_improved.xml',
                        help='Path to LBP cascade XML file.')
    parser.add_argument('--width', type=int, default=224, help='Resized width')
    parser.add_argument('--height', type=int, default=224, help='Resized height')
    parser.add_argument('--keep_placeholder', action='store_true',
                        help='Save a black image if no face is detected.')
    args = parser.parse_args()

    preprocess_sequences(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cascade_path=args.cascade,
        output_size=(args.width, args.height),
        keep_placeholder=args.keep_placeholder
    )

if __name__ == '__main__':
    main()
