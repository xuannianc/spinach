from keras.applications import InceptionV3, imagenet_utils
from keras.models import Model
from dogs_vs_cats.config import DATASET_DIR
import os.path as osp
import cv2
import numpy as np


inception_v3 = InceptionV3(weights='imagenet')
inception_v3.summary()
model = Model(input=inception_v3.input, output=inception_v3.get_layer('avg_pool').output)
image1_path = osp.join(DATASET_DIR, 'train', 'cat.1000.jpg')
image1 = cv2.imread(image1_path)
image1 = np.expand_dims(image1, axis=0)
image1 = imagenet_utils.preprocess_input(image1)
image2_path = osp.join(DATASET_DIR, 'train', 'cat.1001.jpg')
image2 = cv2.imread(image2_path)
image2 = np.expand_dims(image2, axis=0)
image2 = imagenet_utils.preprocess_input(image2)
# input = np.vstack((image1, image2))
feature1 = model.predict(image1)[0]
print(feature1.shape)
feature2 = model.predict(image2)[0]
print('feature1={}'.format(feature1))
print('feature2={}'.format(feature2))
print(np.linalg.norm(feature1 - feature2))