import tensorflow as tf
from preprocess import get_data_generators
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

model = tf.keras.models.load_model(config['model_save_path'])
_, val_gen = get_data_generators(config)

results = model.evaluate(val_gen)
print("Validation Results:", results)
