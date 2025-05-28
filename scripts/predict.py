import tensorflow as tf
import numpy as np
import argparse
import yaml
import cv2
import os

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load class names
quality_classes = config['quality_classes']
region_classes = config['region_classes']

# Load model
model = tf.keras.models.load_model(config['model_save_path'])

# Image preprocessing function
def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at: {image_path}")
    image = cv2.resize(image, tuple(image_size))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# Prediction function
def predict(image_path):
    image = preprocess_image(image_path, config['image_size'])
    quality_pred, region_pred = model.predict(image)

    quality_index = np.argmax(quality_pred)
    region_index = np.argmax(region_pred)

    quality = quality_classes[quality_index]
    region = region_classes[region_index]

    print(f"\n🧪 Prediction Result for: {os.path.basename(image_path)}")
    print(f"👉 Tea Quality: {quality}")
    print(f"👉 Tea Region : {region}\n")

# CLI setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input tea liquor image')
    args = parser.parse_args()

    predict(args.image_path)
