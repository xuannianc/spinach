# import the necessary packages
from emotion_recognition.config import emotion_config as config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from hdf5.hdf5databasegenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", type=str,
#                 help="path to model checkpoint to load")
# args = vars(ap.parse_args())
model_checkpoint = 'hdf5/lowest_loss_model.hdf5'

# initialize the testing data generator and image preprocessor
test_aug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()
# initialize the testing dataset generator
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
                                aug=test_aug, preprocessors=[iap], num_classes=config.NUM_CLASSES)
# load the model from disk
print("[INFO] loading {}...".format(model_checkpoint))
model = load_model(model_checkpoint)
# evaluate the network
(loss, acc) = model.evaluate_generator(
    test_gen.generator(),
    steps=test_gen.num_images // config.BATCH_SIZE,
    max_queue_size=config.BATCH_SIZE * 2)
print("[INFO] accuracy: {:.2f}".format(acc * 100))
# close the testing database
test_gen.close()
