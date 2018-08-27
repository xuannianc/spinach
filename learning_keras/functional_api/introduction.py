from keras import Input, layers
from keras.models import Sequential, Model
# A tensor
input_tensor = Input(shape=(32,))
# A layer is a function.
dense = layers.Dense(32, activation='relu')
# A layer may be called on a tensor, and it returns a tensor.
output_tensor = dense(input_tensor)
# Sequential model, which you already know about
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))
# Its functional equivalent
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
# The Model class turns an input tensor and output tensor into a model.
model = Model(input_tensor, output_tensor)
model.summary()
# Compiles the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
import numpy as np
# Generates dummy Numpy data to train on
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))
# Trains the model for 10 epochs
model.fit(x_train, y_train, epochs=10, batch_size=128)
# Evaluates the model
score = model.evaluate(x_train, y_train)
