from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import LabelBinarizer

BATCH_SIZE = 128
NB_EPOCH = 40
VERBOSE = 1
# OPTIM = RMSprop()
OPTIM = SGD(lr=1e-1)
# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float")
X_test = X_test.astype("float")
X_train /= 255
X_test /= 255
# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
# augumenting
print("Augmenting training set images...")
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
# generate batch_size samples
# f = datagen.flow(X_train, batch_size=10,
#                  save_to_dir='preview', save_prefix='cifar', save_format='jpeg')
# fit the dataget
datagen.fit(X_train)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()
# compile
model.compile(loss='categorical_crossentropy', optimizer=OPTIM,
              metrics=['accuracy'])
# train
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                              validation_data=(X_test, y_test),
                              steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                              epochs=NB_EPOCH, verbose=VERBOSE)
# evaluate
score = model.evaluate(X_test, y_test,
                       batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])
