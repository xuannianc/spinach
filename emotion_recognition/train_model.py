# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")
# import the necessary packages
from emotion_recognition.config import emotion_config as config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
# from callbacks import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
from hdf5.hdf5databasegenerator import HDF5DatasetGenerator
from emotion_recognition.emotionvggnet import EmotionVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import os
from keras.callbacks import ModelCheckpoint

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--checkpoints", required=True,
#                 help="path to output checkpoint directory")
# ap.add_argument("-m", "--model", type=str,
#                 help="path to *specific* model checkpoint to load")
# ap.add_argument("-s", "--start-epoch", type=int, default=0,
#                 help="epoch to restart training at")
# args = vars(ap.parse_args())
model_checkpoint = 'hdf5/lowest_loss_model.hdf5'
# construct the training and testing image generators for data
# augmentation, then initialize the image preprocessor
train_aug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                               horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
val_aug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()
# initialize the training and validation dataset generators
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
                                 aug=train_aug, preprocessors=[iap], num_classes=config.NUM_CLASSES)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
                               aug=val_aug, preprocessors=[iap], num_classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
print("[INFO] compiling model...")
model = EmotionVGGNet.build(width=48, height=48, depth=1,
                            num_classes=config.NUM_CLASSES)
opt = Adam(lr=1e-3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# construct the set of callbacks
callbacks = [
    ModelCheckpoint(model_checkpoint, monitor='val_loss', save_best_only=True, verbose=1)]
# TrainingMonitor(fig_path, jsonPath=json_path,
#                 startAt=args["start_epoch"])]
# train the network
model.fit_generator(
    train_gen.generator(),
    steps_per_epoch=train_gen.num_images // config.BATCH_SIZE,
    validation_data=val_gen.generator(),
    validation_steps=val_gen.num_images // config.BATCH_SIZE,
    epochs=15,
    max_queue_size=config.BATCH_SIZE * 2,
    callbacks=callbacks, verbose=1)
# close the databases
train_gen.close()
val_gen.close()
