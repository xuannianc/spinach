from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Embedding

# Number of words to consider as features
max_features = 10000
# Cuts off the text after this number of words (among the max_features most common words)
maxlen = 20
# Loads the data as lists of integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train[0][:10])
# Turns the lists of integers into a 2D integer tensor of shape (samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_train[0][:10])

model = Sequential()
# Specifies the maximum input length to the
# Embedding layer so you can later flatten the
# embedded inputs. After the Embedding layer,
# the activations have shape (samples, maxlen, 8).
model.add(Embedding(10000, 8, input_length=maxlen))
# Flattens the 3D tensor of embeddings into a 2D
# tensor of shape (samples, maxlen * 8)
model.add(Flatten())
# Adds the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
