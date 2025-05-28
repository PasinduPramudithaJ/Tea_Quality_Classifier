# tea_cnn_model.py
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_tea_quality_model(input_shape=(128, 128, 3), num_quality_classes=3, num_region_classes=7):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)

    quality_output = Dense(64, activation='relu')(x)
    quality_output = Dense(num_quality_classes, activation='softmax', name='quality')(quality_output)

    region_output = Dense(64, activation='relu')(x)
    region_output = Dense(num_region_classes, activation='softmax', name='region')(region_output)

    model = Model(inputs=inputs, outputs=[quality_output, region_output])
    return model
