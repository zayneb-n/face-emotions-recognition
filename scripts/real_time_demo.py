# scripts/real_time_demo.py

import cv2
import numpy as np
import argparse
import joblib
import json
import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

def load_models(pca_model_path, kmeans_model_path, classifier_path, label_encoder_path, cluster_mapping_path):
    """
    Load PCA, K-Means, Classifier, Label Encoder, and Cluster Mapping models.
    """
    print("Loading PCA model...")
    pca = joblib.load(pca_model_path)
    
    print("Loading K-Means model...")
    kmeans = joblib.load(kmeans_model_path)
    
    print("Loading Emotion Classifier...")
    classifier = joblib.load(classifier_path)
    
    print("Loading Label Encoder...")
    label_encoder = joblib.load(label_encoder_path)
    
    print("Loading Cluster Mapping...")
    with open(cluster_mapping_path, 'r', encoding='utf-8') as f:
        cluster_mapping = json.load(f)
    
    return pca, kmeans, classifier, label_encoder, cluster_mapping

def load_feature_model():
    """
    Load the pretrained MobileNetV2 model for feature extraction.
    """
    print("Loading MobileNetV2 model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return base_model

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess the frame for feature extraction.
    """
    img_resized = cv2.resize(frame, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype(np.float32)
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

def main():
    parser = argparse.ArgumentParser(description="Real-Time Facial Emotion Recognition Demo")
    parser.add_argument('--pca_model', type=str, default='models/pca_model.joblib',
                        help='Path to the PCA model.')
    parser.add_argument('--kmeans_model', type=str, default='models/kmeans_model.joblib',
                        help='Path to the K-Means model.')
    parser.add_argument('--classifier', type=str, default='models/emotion_classifier.joblib',
                        help='Path to the trained classifier.')
    parser.add_argument('--label_encoder', type=str, default='models/label_encoder.joblib',
                        help='Path to the label encoder.')
    parser.add_argument('--cluster_mapping', type=str, default='models/cluster_mapping.json',
                        help='Path to the cluster-to-emotion mapping JSON.')
    parser.add_argument('--cascade', type=str, default='scripts/lbpcascade_frontalface_improved.xml',
                        help='Path to the face detection cascade XML file.')
    args = parser.parse_args()

    # Verify all model paths exist
    required_files = [args.pca_model, args.kmeans_model, args.classifier, args.label_encoder, args.cluster_mapping, args.cascade]
    for path in required_files:
        if not os.path.exists(path):
            print(f"Error: The file {path} does not exist.")
            sys.exit(1)

    # Load models and mappings
    pca, kmeans, classifier, label_encoder, cluster_mapping = load_models(
        pca_model_path=args.pca_model,
        kmeans_model_path=args.kmeans_model,
        classifier_path=args.classifier,
        label_encoder_path=args.label_encoder,
        cluster_mapping_path=args.cluster_mapping
    )

    # Load feature extraction model
    feature_model = load_feature_model()

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(args.cascade)
    if face_cascade.empty():
        print(f"Error: Cannot load cascade file at {args.cascade}")
        sys.exit(1)

    # Initialize webcam
    print("Starting webcam for real-time emotion recognition...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess face
            face_preprocessed = preprocess_frame(face_roi)
            face_batch = np.expand_dims(face_preprocessed, axis=0)  # Shape: (1, 224, 224, 3)

            # Extract features
            features = feature_model.predict(face_batch)  # Shape: (1, 1280)

            # Apply PCA
            reduced_features = pca.transform(features)  # Shape: (1, 50)

            # Predict cluster
            cluster_id = kmeans.predict(reduced_features)[0]

            # Predict emotion using classifier
            emotion_pred = classifier.predict(reduced_features)[0]
            emotion_label = label_encoder.inverse_transform([emotion_pred])[0]

            # Overlay emotion label on the frame
            cv2.putText(frame, f"{emotion_label}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the resulting frame
        cv2.imshow('Real-Time Facial Emotion Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
