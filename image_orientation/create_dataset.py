# import the necessary packages
from imutils import paths
import numpy as np
import progressbar
import argparse
import imutils
import random
import cv2
import os

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# help="path to input directory of images")
# ap.add_argument("-o", "--output", required=True,
# help="path to output directory of rotated iamges")
# args = vars(ap.parse_args())
dataset = '/home/adam/.keras/datasets/indoor_cvpr/images'
output = '/home/adam/.keras/datasets/indoor_cvpr/rotated_images'
# grab the paths to the input images (limiting ourselves to 10,000
# images) and shuffle them to make creating a training and testing
# split easier
image_paths = list(paths.list_images(dataset))[:10000]
random.shuffle(image_paths)
# initialize a dictionary to keep track of the number of each angle
# chosen so far, then initialize the progress bar
counts = {}
widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths),
                               widgets=widgets).start()
# loop over the image paths
for (i, image_path) in enumerate(image_paths):
    # determine the rotation angle, and load the image
    angle = np.random.choice([0, 90, 180, 270])
    image = cv2.imread(image_path)
    # if the image is None (meaning there was an issue loading the
    # image from disk, simply skip it)
    if image is None:
        continue
    # rotate the image based on the selected angle, then construct
    # the path to the base output directory
    image = imutils.rotate_bound(image, angle)
    base = os.path.sep.join([output, str(angle)])
    # if the base path does not exist already, create it
    if not os.path.exists(base):
        os.makedirs(base)
    # extract the image file extension, then construct the full path
    # to the output file
    ext = image_path[image_path.rfind("."):]
    output_path = [base, "image_{}{}".format(
        str(counts.get(angle, 0)).zfill(5), ext)]
    output_path = os.path.sep.join(output_path)
    # save the image
    cv2.imwrite(output_path, image)
    # update the count for the angle
    c = counts.get(angle, 0)
    counts[angle] = c + 1
    pbar.update(i)
# finish the progress bar
pbar.finish()
# loop over the angles and display counts for each of them
for angle in sorted(counts.keys()):
    print("[INFO] angle={}: {:,}".format(angle, counts[angle]))
