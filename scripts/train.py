import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from models.tea_cnn_model import build_tea_quality_model
from preprocess import get_data_generators

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

input_shape = tuple(config['image_size']) + (3,)
model = build_tea_quality_model(
    input_shape=input_shape,
    num_quality_classes=config['num_quality_classes'],
    num_region_classes=config['num_region_classes']
)

model.compile(
    optimizer='adam',
    loss={'quality': 'categorical_crossentropy', 'region': 'categorical_crossentropy'},
    metrics={'quality': 'accuracy', 'region': 'accuracy'}
)

train_gen, val_gen = get_data_generators(config)

checkpoint = ModelCheckpoint(config['model_save_path'], monitor='val_loss', save_best_only=True)
model.fit(train_gen,
          validation_data=val_gen,
          epochs=config['epochs'],
          callbacks=[checkpoint])
