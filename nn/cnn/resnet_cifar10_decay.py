# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from nn.cnn.resnet import ResNet
from callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import sys
import os

# define the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 100
INIT_LR = 1e-1


def poly_decay(epoch, lr):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    max_epochs = NUM_EPOCHS
    base_lr = INIT_LR
    power = 1.0
    # compute the new learning rate based on polynomial decay
    new_lr = base_lr * (1 - (epoch / float(max_epochs))) ** power
    # return the new learning rate
    return new_lr


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
#                 help="path to output model")
# ap.add_argument("-o", "--output", required=True,
#                 help="path to output directory (logs, plots, etc.)")
# args = vars(ap.parse_args())
MODEL_PATH = 'checkpoints/resnet_100_decay.hdf5'

# load the training and testing data, converting the images from
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
x_train = x_train.astype("float")
x_test = x_test.astype("float")
# apply mean subtraction to the data
# shape 为 (32,32,3)
mean = np.mean(x_train, axis=0)
x_train -= mean
x_test -= mean
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(y_train)
testY = lb.transform(y_test)
# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")
# construct the set of callbacks
callbacks = [TrainingMonitor(fig_path='resnet56_cifar10.jpg', json_path='resnet56_cifar10.json'),
             LearningRateScheduler(poly_decay)]
# initialize the optimizer and model (ResNet-56)
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
                     (64, 64, 128, 256), regularization=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the network
print("[INFO] training network...")
model.fit_generator(
    aug.flow(x_train, y_train, batch_size=128),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // 128, epochs=NUM_EPOCHS,
    callbacks=callbacks, verbose=1)
# save the network to disk
print("[INFO] serializing network...")
model.save(MODEL_PATH)
