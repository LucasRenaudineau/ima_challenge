import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Input

# Import files
from load import *
from data_augmentation import *

if __name__ == "__main__":
    base_model = keras.applications.EfficientNetV2S( # looks like a reasonable size for good performances
        weights='imagenet',
        input_shape=(368, 368, 3),
        include_top=False)
    
    inputs = keras.Input(shape=(368, 368, 3))
    x = data_augmentation(inputs)
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1) # requires scaled inputs between -1 and 1
    x = scale_layer(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(13)(x)
    model = keras.Model(inputs, outputs)
    
    model.summary()
    
    model.save("./outputs/model_not_trained.keras")
