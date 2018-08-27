# import the necessary packages
import os

# initialize the base path for the LISA dataset
DATASET_PATH = "/home/adam/.keras/datasets/lisa"
# build the path to the annotations file
ANNOT_PATH = os.path.sep.join([DATASET_PATH, "allAnnotations.csv"])
# build the path to the output training and testing record files,
# along with the class labels file
TRAIN_RECORD = "records/training.record"
TEST_RECORD = "records/testing.record"
CLASSES_FILE = "records/classes.pbtxt"
# initialize the test split size
TEST_SIZE = 0.25
# initialize the class labels dictionary
CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}
# CLASSES = {1: "pedestrianCrossing", 2: "signalAhead", 3: "stop"}
