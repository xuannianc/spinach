# import the necessary packages
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import h5py
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--db", required=True,
#                 help="path HDF5 database")
# ap.add_argument("-i", "--dataset", required=True,
#                 help="path to the input images dataset")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to trained orientation model")
# args = vars(ap.parse_args())
database = 'hdf5/data.hdf5'
dataset = '/home/adam/.keras/datasets/indoor_cvpr/rotated_images'
model = 'model.cpickle'
# load the label names (i.e., angles) from the HDF5 dataset
db = h5py.File(database)
label_names = [int(angle) for angle in db["label_names"][:]]
db.close()
# grab the paths to the testing images and randomly sample them
print("[INFO] sampling images...")
image_paths = list(paths.list_images(dataset))
image_paths = np.random.choice(image_paths, size=(10,), replace=False)
# load the VGG16 network
print("[INFO] loading network...")
vgg = VGG16(weights="imagenet", include_top=False)
# load the orientation model
print("[INFO] loading model...")
model = pickle.loads(open("model.cpickle", "rb").read())
# loop over the image paths
for image_path in image_paths:
    # load the image via OpenCV so we can manipulate it after
    # classification
    orig = cv2.imread(image_path)
    # load the input image using the Keras helper utility while
    # ensuring the image is resized to 224x224 pixels
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    # preprocess the image by (1) expanding the dimensions and (2)
    # subtracting the mean RGB pixel intensity from the ImageNet
    # dataset
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    # pass the image through the network to obtain the feature vector
    features = vgg.predict(image)
    features = features.reshape((features.shape[0], 512 * 7 * 7))
    # now that we have the CNN features, pass these through our
    # classifier to obtain the orientation predictions
    angle = model.predict(features)
    angle = label_names[angle[0]]
    # now that we have the predicted orientation of the image we can
    # correct for it
    rotated = imutils.rotate_bound(orig, 360 - angle)
    # display the original and corrected images
    cv2.imshow("Original", orig)
    cv2.imshow("Corrected", rotated)
    cv2.waitKey(0)
