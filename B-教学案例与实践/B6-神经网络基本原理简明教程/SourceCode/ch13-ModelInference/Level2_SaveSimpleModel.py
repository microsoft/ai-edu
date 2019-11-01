from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='input_data')
dense_result = layers.Dense(64)(inputs)
outputs = layers.ReLU()(dense_result)

model = keras.Model(inputs=inputs, outputs=outputs, name='simple_model')
model.save('simple_model.h5')

