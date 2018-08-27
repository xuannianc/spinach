# import the necessary packages
from emotion_recognition.config import emotion_config as config
from hdf5.hdf5databasewriter import HDF5DatasetWriter
import numpy as np

# open the input file for reading (skipping the header), then
# initialize the list of data and labels for the training,
# validation, and testing sets
print("[INFO] loading input data...")
f = open(config.INPUT_PATH)
f.__next__()  # f.next() for Python 2.7
(train_images, train_labels) = ([], [])
(val_images, val_labels) = ([], [])
(test_images, test_labels) = ([], [])
# loop over the rows in the input file
for row in f:
    # extract the label, image, and usage from the row
    (label, image, usage) = row.strip().split(",")
    label = int(label)
    # if we are ignoring the "disgust" class there will be 6 total
    # class labels instead of 7
    if config.NUM_CLASSES == 6:
        # merge together the "anger" and "disgust classes
        if label == 1:
            label = 0
        # if label has a value greater than zero, subtract one from
        # it to make all labels sequential (not required, but helps
        # when interpreting results)
        if label > 0:
            label -= 1
    # reshape the flattened pixel list into a 48x48 (grayscale)
    # image
    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))
    # check if we are examining a training image
    if usage == "Training":
        train_images.append(image)
        train_labels.append(label)
    # check if this is a validation image
    elif usage == "PrivateTest":
        val_images.append(image)
        val_labels.append(label)
    # otherwise, this must be a testing image
    else:
        test_images.append(image)
        test_labels.append(label)
# construct a list pairing the training, validation, and testing
# images along with their corresponding labels and output HDF5
# files
datasets = [
    (train_images, train_labels, config.TRAIN_HDF5),
    (val_images, val_labels, config.VAL_HDF5),
    (test_images, test_labels, config.TEST_HDF5)]

# loop over the dataset tuples
for (images, labels, output_path) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(output_path))
    writer = HDF5DatasetWriter((len(images), 48, 48), output_path)
    # loop over the image and add them to the dataset
    for (image, label) in zip(images, labels):
        writer.add([image], [label])
    # close the HDF5 writer
    writer.close()
# close the input file
f.close()
