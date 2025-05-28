import numpy as np
import tensorflow as tf

def encode_labels(labels, class_names):
    class_dict = {name: idx for idx, name in enumerate(class_names)}
    encoded = [class_dict[label] for label in labels]
    return tf.keras.utils.to_categorical(encoded, num_classes=len(class_names))
