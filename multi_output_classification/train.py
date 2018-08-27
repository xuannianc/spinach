# USAGE
# python train.py --dataset dataset --model output/fashion.model \
#	--categorybin output/category_lb.pickle --colorbin output/color_lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from multi_output_classification.fashionnet import FashionNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset (i.e., directory of images)")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model")
# ap.add_argument("-l", "--categorybin", required=True,
# 	help="path to output category label binarizer")
# ap.add_argument("-c", "--colorbin", required=True,
# 	help="path to output color label binarizer")
# ap.add_argument("-p", "--plot", type=str, default="output",
# 	help="base filename for generated plots")
# args = vars(ap.parse_args())
DATASET_DIR = 'dataset'
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
image_paths = sorted(list(paths.list_images(DATASET_DIR)))
random.seed(42)
random.shuffle(image_paths)

# initialize the data, clothing category labels (i.e., shirts, jeans,
# dresses, etc.) along with the color labels (i.e., red, blue, etc.)
data = []
category_labels = []
color_labels = []

# loop over the input images
for image_path in image_paths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = img_to_array(image)
    data.append(image)

    # extract the clothing color and category from the path and
    # update the respective lists
    (color, category) = image_path.split(os.path.sep)[-2].split("_")
    category_labels.append(category)
    color_labels.append(color)

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
    len(image_paths), data.nbytes / (1024 * 1000.0)))

# convert the label lists to NumPy arrays prior to binarization
category_labels = np.array(category_labels)
color_labels = np.array(color_labels)

# binarize both sets of labels
print("[INFO] binarizing labels...")
category_lb = LabelBinarizer()
color_lb = LabelBinarizer()
category_labels = category_lb.fit_transform(category_labels)
color_labels = color_lb.fit_transform(color_labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, category_labels, color_labels,
                         test_size=0.2, random_state=42)
(train_x, test_x, train_category_y, test_category_y,
 train_color_y, test_color_y) = split

# initialize our FashionNet multi-output network
model = FashionNet.build(96, 96,
                         num_categories=len(category_lb.classes_),
                         num_colors=len(color_lb.classes_),
                         final_act="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
# 字典的 key 和 model 最后一个 activation layer 的 name 保持一致
losses = {
    "category_output": "categorical_crossentropy",
    "color_output": "categorical_crossentropy",
}
loss_weights = {"category_output": 1.0, "color_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights,
              metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(train_x,
              {"category_output": train_category_y, "color_output": train_color_y},
              validation_data=(test_x, {"category_output": test_category_y, "color_output": test_color_y}),
              epochs=EPOCHS,
              verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("model.hdf5")

# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open("category_label_names.pickle", "wb")
f.write(pickle.dumps(category_lb))
f.close()

# save the color binarizer to disk
print("[INFO] serializing color label binarizer...")
f = open("color_label_names.pickle", "wb")
f.write(pickle.dumps(color_lb))
f.close()

# plot the total loss, category loss, and color loss
loss_names = ["loss", "category_output_loss", "color_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(loss_names):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
    ax[i].legend()

# save the losses figure
plt.tight_layout()
plt.savefig("losses.jpg")
plt.close()

# create a new figure for the accuracies
accuracy_names = ["category_output_acc", "color_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

# loop over the accuracy names
for (i, l) in enumerate(accuracy_names):
    # plot the loss for both the training and validation data
    ax[i].set_title("Accuracy for {}".format(l))
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("accs.jpg")
plt.close()
