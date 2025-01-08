import os
import argparse
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Loads an image from disk, converts BGR to RGB, resizes, then applies
    MobileNetV2 preprocess_input. Returns a NumPy array of shape (224, 224, 3).
    """
    img_bgr = cv2.imread(img_path)  # shape: (H, W, 3) in BGR
    if img_bgr is None:
        return None  # Could not load image for some reason
    
    # Resize to match model's expected input (224x224 for MobileNetV2)
    img_bgr = cv2.resize(img_bgr, target_size)

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to float32 and preprocess
    # MobileNetV2 expects inputs in range [-1,1] after preprocess_input
    x = img_rgb.astype(np.float32)
    x = preprocess_input(x)  # from tensorflow.keras.applications.mobilenet_v2
    return x

def create_image_list(input_dir):
    """
    Traverse the input_dir and create a list of all .png (or .jpg) image paths.
    Returns a list of absolute file paths.
    """
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith('.png'):  # or .jpg
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)
    return sorted(image_paths)

def batch_generator(image_paths, batch_size=16, target_size=(224, 224)):
    """
    Yields batches of preprocessed images (NumPy array) and the corresponding paths.
    """
    n = len(image_paths)
    idx = 0
    while idx < n:
        batch_paths = image_paths[idx: idx + batch_size]
        
        batch_imgs = []
        valid_paths = []  # might skip images that can't be loaded
        for p in batch_paths:
            arr = load_and_preprocess_image(p, target_size=target_size)
            if arr is not None:
                batch_imgs.append(arr)
                valid_paths.append(p)
        
        if len(batch_imgs) == 0:
            # means none in this batch could be loaded
            idx += batch_size
            continue
        
        batch_imgs = np.stack(batch_imgs, axis=0)  # shape: (batch_size, 224, 224, 3)
        yield batch_imgs, valid_paths
        
        idx += batch_size

def extract_features(
    input_dir, output_npy, output_txt=None,
    batch_size=16, target_size=(224,224)
):
    """
    1. Collect all images in input_dir.
    2. Load MobileNetV2 pretrained model (without top).
    3. For each batch of images, predict embeddings.
    4. Concatenate embeddings into a big array, save to .npy
    5. Optionally, save a .txt or .json mapping row -> image path.
    """
    print(f"Collecting images from: {input_dir}")
    image_paths = create_image_list(input_dir)
    print(f"Found {len(image_paths)} images.")

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    # The output shape from MobileNetV2 with pooling='avg' is (None, 1280)

    all_embeddings = []
    all_paths = []
    
    # Generate batches
    gen = batch_generator(image_paths, batch_size=batch_size, target_size=target_size)
    
    total_processed = 0
    for batch_imgs, valid_paths in gen:
        # Predict embeddings
        embeddings = base_model.predict(batch_imgs)  # shape: (batch_size, 1280)
        all_embeddings.append(embeddings)
        all_paths.extend(valid_paths)
        
        total_processed += len(valid_paths)
        print(f"Processed {total_processed} / {len(image_paths)} images", end='\r')
    
    print()  # newline
    # Concatenate all
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # shape: (N, 1280)

    # Save embeddings
    np.save(output_npy, all_embeddings)
    print(f"Embeddings saved to {output_npy}, shape={all_embeddings.shape}")

    # Optionally save the mapping (row index -> image path)
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as f:
            for i, p in enumerate(all_paths):
                f.write(f"{i}\t{p}\n")
        print(f"Mapping of embeddings to image paths saved at {output_txt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='Dataset/AFEW-VA-processed',
                        help='Directory with cropped face images.')
    parser.add_argument('--output_npy', type=str, default='features.npy',
                        help='File path to save the embeddings (npy).')
    parser.add_argument('--output_txt', type=str, default='paths.txt',
                        help='Optional file to save the mapping between row index and image path.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference.')
    parser.add_argument('--width', type=int, default=224, help='Image width for resizing.')
    parser.add_argument('--height', type=int, default=224, help='Image height for resizing.')
    args = parser.parse_args()

    extract_features(
        input_dir=args.input_dir,
        output_npy=args.output_npy,
        output_txt=args.output_txt,
        batch_size=args.batch_size,
        target_size=(args.width, args.height)
    )

if __name__ == '__main__':
    main()
