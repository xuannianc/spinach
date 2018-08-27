# import the necessary packages
from os import path

# define the base path to the cars dataset
DATASETS_PATH = "/home/adam/.keras/datasets/standford_car"
# based on the base path, derive the images path and meta file path
IMAGES_PATH = path.sep.join([DATASETS_PATH, "car_ims"])
LABELS_PATH = path.sep.join([DATASETS_PATH, "complete_dataset.csv"])
# define the path to the output training, validation, and testing hdf5 file
TRAIN_HDF5 = 'hdf5/train.hdf5'
VAL_HDF5 = 'hdf5/val.hdf5'
TEST_HDF5 = 'hdf5/test.hdf5'
# define the path to the label encoder
LABEL_ENCODER_PATH = "output/le.cpickle"
# define the percentage of validation and testing images relative
# to the number of training images
NUM_CLASSES = 164
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15
# define the batch size
BATCH_SIZE = 32
