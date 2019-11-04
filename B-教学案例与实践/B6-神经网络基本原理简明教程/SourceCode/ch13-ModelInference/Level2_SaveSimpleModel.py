from tensorflow import keras

inputs = keras.Input(shape=(784,), name='input_data')
dense_result = keras.layers.Dense(10)(inputs)
outputs = keras.layers.ReLU()(dense_result)

model = keras.Model(inputs=inputs, outputs=outputs, name='simple_model')
model.save('simple_model.h5')

