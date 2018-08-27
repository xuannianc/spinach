# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from nn.cnn.resnet import ResNet
from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import sys

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--checkpoints", required=True,
#                 help="path to output checkpoint directory")
# ap.add_argument("-m", "--model", type=str,
#                 help="path to *specific* model checkpoint to load")
# # 用于保存模型的 name 后面的序号
# ap.add_argument("-s", "--start-epoch", type=int, default=0,
#                 help="epoch to restart training at")
# args = vars(ap.parse_args())

CHECKPOINTS_DIR = 'checkpoints'
MODEL_PATH = 'checkpoints/epoch_50.hdf5'
START_EPOCH = 50

# load the training and testing data, converting the images from
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
x_train = x_train.astype("float")
x_test = x_test.astype("float")
# apply mean subtraction to the data
mean = np.mean(x_train, axis=0)
x_train -= mean
x_test -= mean
# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")
# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
if MODEL_PATH:
    print("[INFO] loading {}...".format(MODEL_PATH))
    model = load_model(MODEL_PATH)
    # update the learning rate
    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-2)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    print("[INFO] compiling model...")
# otherwise, load the checkpoint from disk
else:
    opt = SGD(lr=1e-1)
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
                         (64, 64, 128, 256), regularization=0.0005)
    model.summary()
    print('[INFO] num of layers: {}'.format(len(model.layers)))
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
# construct the set of callbacks
callbacks = [
    EpochCheckpoint(CHECKPOINTS_DIR, every=5,
                    start_at=START_EPOCH),
    TrainingMonitor(fig_path="resnet56_cifar10.png",
                    json_path="resnet56_cifar10.json",
                    start_at=START_EPOCH)]
# train the network
print("[INFO] training network...")
model.fit_generator(
    aug.flow(x_train, y_train, batch_size=128),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // 128, epochs=10,
    callbacks=callbacks, verbose=1)
