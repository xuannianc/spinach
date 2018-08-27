from keras import layers
from keras import Input
from keras.models import Model

# Instantiates a single LSTM layer, once
lstm = layers.LSTM(32)
# Building the left branch of the model: inputs are variable-length
# sequences of vectors of size 128.
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)
# Building the right branch of the model: when you call an existing layer
# instance, you reuse its weights.
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)
# Builds the classifier on top
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)
# Instantiating and training the model: when you
# train such a model, the weights of the LSTM layer
# are updated based on both inputs.
model = Model([left_input, right_input], predictions)
model.fit([left_data, right_data], targets)
