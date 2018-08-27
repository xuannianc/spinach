import numpy as np
import os
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from dogs_vs_cats.config import DATASET_DIR
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

data_dir = osp.join(DATASET_DIR, 'data')
image_height, image_width = 150, 150
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')
num_classes = 2
epochs = 50
batch_size = 20
num_train = 2000
num_val = 1000
input_shape = (image_height, image_width, 3)
train_epoch_steps = num_train // batch_size
val_epoch_steps = num_val // batch_size

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

vgg = keras.applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=input_shape)


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    train_generator = train_datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = vgg.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
val_features, val_labels = extract_features(val_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
val_features = np.reshape(val_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss=losses.binary_crossentropy,
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

history = model.fit(
    train_features,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_features, val_labels)
)
# plot the training loss and accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
