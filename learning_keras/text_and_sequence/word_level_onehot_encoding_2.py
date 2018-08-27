from keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# creates a tokenizer,configured to only take into account the 1,000 most common words
tokenizer = Tokenizer(num_words=1000)
# builds the word index
tokenizer.fit_on_texts(samples)
# turns strings to lists of indices
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)
# You could also directly get the one-hot
# binary representations. Vectorization
# modes other than one-hot encoding
# are supported by this tokenizer.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print(one_hot_results)
word_index = tokenizer.word_index
print(word_index)
# How you can recover the word index that was computed
print('Found %s unique tokens.' % len(word_index))
