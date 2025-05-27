import tensorflow as tf
import os

def get_data_generators(config):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    def create_generator(directory):
        return datagen.flow_from_directory(
            directory,
            target_size=tuple(config['image_size']),
            batch_size=config['batch_size'],
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

    # Use a custom folder naming format: data/train/region/quality/images.jpg
    def custom_generator(path):
        base_gen = datagen.flow_from_directory(
            path,
            target_size=tuple(config['image_size']),
            batch_size=config['batch_size'],
            class_mode=None,
            shuffle=True,
            seed=42
        )
        while True:
            batch = next(base_gen)
            quality_labels = []
            region_labels = []

            for filepath in base_gen.filepaths:
                region = filepath.split(os.sep)[-3]
                quality = filepath.split(os.sep)[-2]
                region_labels.append(region)
                quality_labels.append(quality)

            # Use helper to one-hot encode
            from utils.helpers import encode_labels
            y_quality = encode_labels(quality_labels, config['quality_classes'])
            y_region = encode_labels(region_labels, config['region_classes'])

            yield batch, {'quality': y_quality, 'region': y_region}

    train_gen = custom_generator(config['train_dir'])
    val_gen = custom_generator(config['val_dir'])

    return train_gen, val_gen
