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
from keras.preprocessing.image import ImageDataGenerator

data_dir = osp.join(DATASET_DIR, 'data')
image_height, image_width = 150, 150
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')
epochs = 50
batch_size = 20
num_train = 2000
num_val = 1000
input_shape = (image_height, image_width, 3)
train_epoch_steps = num_train // batch_size
val_epoch_steps = num_val // batch_size

vgg = keras.applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=input_shape)

model = models.Sequential()
model.add(vgg)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())
print('This is the number of trainable weights before freezing the vgg: {}'.format(len(model.trainable_weights)))
vgg.trainable = False
print('This is the number of trainable weights after freezing the vgg: {}'.format(len(model.trainable_weights)))
model.compile(loss=losses.binary_crossentropy,
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Note that the validation data shouldn’t be augmented!
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    # Target directory
    target_size=(150, 150),
    # Resizes all images to 150 × 150
    batch_size=20,
    # Because you use binary_crossentropy loss, you need binary labels.
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_epoch_steps,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_epoch_steps
)
# plot the training loss and accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
        return smoothed_points


plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
