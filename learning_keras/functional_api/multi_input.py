from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
# The text input is a variable-length sequence of integers.
# Note that you can optionally name the inputs.
text_input = Input(shape=(None,), dtype='int32', name='text')
# Embeds the inputs into a sequence of vectors of size 64
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
# Encodes the vectors in a single vector via an LSTM
encoded_text = layers.LSTM(32)(embedded_text)
# Same process (with different layer instances) for the question
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
# Concatenates the encoded question and encoded text
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
# Adds a softmax classifier on top
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)
# At model instantiation, you specify the two inputs and the output.
model = Model([text_input, question_input], answer)
model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
import numpy as np

num_samples = 1000
max_length = 100
# Generates dummy Numpy data
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
# Answers are one-hot encoded, not integers
answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))
# Fitting using a list of inputs
model.fit([text, question], answers, epochs=10, batch_size=128)
# Fitting using a dictionary of inputs (only if inputs are named)
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)
