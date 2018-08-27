import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from keras import models

DATASETS = '/home/adam/.keras/datasets/kaggle_dogs_vs_cats'
img_path = osp.join(DATASETS, 'data', 'test', 'cat', 'cat.1700.jpg')
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
# Preprocesses the image into a 4D tensor
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs that were preprocessed this way.
img_tensor /= 255.
# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)
plt.imshow(img_tensor[0])
plt.show()
simple_cnn_model = models.load_model('../../../dogs_vs_cats/simple_cnn_2.h5')
# Extracts the outputs of the top eight layers
layer_outputs = [layer.output for layer in simple_cnn_model.layers[:8]]
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=simple_cnn_model.input, outputs=layer_outputs)
# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)
print(len(activations))
first_layer_activation = activations[0]
print(first_layer_activation.shape)
# Visualize fourth channel of the activation of the first layer on the test cat picture
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# Visualize seventh channel of the activation of the first layer on the test cat picture
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()





