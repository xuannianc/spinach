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
simple_cnn_model = models.load_model('../../../dogs_vs_cats/simple_cnn_2.h5')
# Extracts the outputs of the top eight layers
layer_outputs = [layer.output for layer in simple_cnn_model.layers[:8]]
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=simple_cnn_model.input, outputs=layer_outputs)
# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)
# Names of the layers, so you can have them as part of your plot
layer_names = []
for layer in simple_cnn_model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    # Number of features in the feature map
    # 就是每一个卷积核的结果是一个 feature,所有 features 组合起来就是 feature map
    n_features = layer_activation.shape[-1]
    # The feature map has shape (1, size, size, n_features).
    size = layer_activation.shape[1]
    n_rows = n_features // images_per_row
    # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_rows, images_per_row * size))
    for row in range(n_rows):
        for col in range(images_per_row):
            channel_image = layer_activation[0, :, :, row * images_per_row + col]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()


