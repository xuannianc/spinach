import keras
from keras import layers
from keras import models
from dogs_vs_cats.config import DATASET_DIR
import os.path as osp
import matplotlib.pyplot as plt

data_dir = osp.join(DATASET_DIR, 'data')
image_height, image_width = 150, 150
train_dir = osp.join(data_dir, 'train')
val_dir = osp.join(data_dir, 'val')
test_dir = osp.join(data_dir, 'test')
num_classes = 2
epochs = 30
batch_size = 20
num_train = 2000
num_val = 1000
num_test = 1000
input_shape = (image_height, image_width, 3)
train_epoch_steps = num_train // batch_size
val_epoch_steps = num_val // batch_size


def simple_cnn(input_shape):
    model = models.Sequential()
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=512, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])
    return model


simple_cnn_model = simple_cnn(input_shape)

# Rescales all images by 1/255
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
# data augmentation
# generator_train = keras.preprocessing.image.ImageDataGenerator(
#     rescale=1. / 255,
#     horizontal_flip=True,
#     zoom_range=0.3,
#     shear_range=0.3
# )

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    # Target directory
    train_dir,
    batch_size=batch_size,
    # Resizes all images to 150 × 150
    target_size=(image_width, image_height),
    # Because you use binary_crossentropy loss, you need binary labels.
    class_mode='binary'
)
# 这个 flow_from_directory 很强大
# 每一次迭代生成一个 tuple,有两个元素
# 第一个元素是一个 batch 的 image data,shape 为 (20,150,150,3)
# 第二个元素是 label
# 如果指定了 class_mode 为 'binary',那么 label 的 shape 为 (20,)
# 如果没有指定 class_mode,默认是 categorical,那么是 one-hot 形式的 label,shape 为 (20,num_classes)
# 而且可以指定 target_size,不需要再做预处理了

# print(next(train_generator)[1].shape)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height),
    class_mode='binary'
)
# print(next(val_generator)[1].shape)

history = simple_cnn_model.fit_generator(
    train_generator,
    steps_per_epoch=train_epoch_steps,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_epoch_steps
)
# model.save('simple_cnn_1.h5')

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

