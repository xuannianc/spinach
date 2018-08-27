# USAGE
# python classify.py --model output/fashion.model \
#	--categorybin output/category_lb.pickle --colorbin output/color_lb.pickle \
#	--image examples/black_dress.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
from imutils import paths

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained model model")
# ap.add_argument("-l", "--categorybin", required=True,
# 	help="path to output category label binarizer")
# ap.add_argument("-c", "--colorbin", required=True,
# 	help="path to output color label binarizer")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

# load the trained convolutional neural network from disk, followed
# by the category and color label binarizers, respectively
print("[INFO] loading network...")
model = load_model("model.hdf5", custom_objects={"tf": tf})
category_lb = pickle.loads(open("category_label_names.pickle", "rb").read())
color_lb = pickle.loads(open("color_label_names.pickle", "rb").read())

# load the image
for image_path in paths.list_images('examples'):
    image = cv2.imread(image_path)
    output = imutils.resize(image, width=400)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    # image = img_to_array(image)
    image = np.expand_dims(image, axis=0)


    # classify the input image using Keras' multi-output functionality
    print("[INFO] classifying image...")
    (category_proba, color_proba) = model.predict(image)

    # find indexes of both the category and color outputs with the
    # largest probabilities, then determine the corresponding class
    # labels
    category_idx = category_proba[0].argmax()
    color_idx = color_proba[0].argmax()
    category_label = category_lb.classes_[category_idx]
    color_label = color_lb.classes_[color_idx]

    # draw the category label and color label on the image
    category_text = "category: {} ({:.2f}%)".format(category_label,
        category_proba[0][category_idx] * 100)
    color_text = "color: {} ({:.2f}%)".format(color_label,
        color_proba[0][color_idx] * 100)
    cv2.putText(output, category_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)
    cv2.putText(output, color_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    print("[INFO] {}".format(category_text))
    print("[INFO] {}".format(color_text))

    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)