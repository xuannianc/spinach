from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import keras.backend as K

# Note that you include the densely connected classifier on top
# in all previous cases, you discarded it.
model = VGG16(weights='imagenet')
print(model.summary())
# Local path to the target image
img_path = 'african_elephant.jpeg'
# Python Imaging Library (PIL) image of size 224 × 224
img = image.load_img(img_path, target_size=(224, 224))
# float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)
# Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)
# Preprocesses the batch (this does channel-wise color normalization)
x = preprocess_input(x)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
print(np.argmax(preds[0]))
# "African elephant" entry in the prediction vector
african_elephant_output = model.output[:, 386]
print(african_elephant_output)
# Output feature map of the block5_conv3 layer,the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')
# Gradient of the 'African elephant' class with regard to the output feature map of block5_conv3
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
print(grads)
# Vector of shape (512,), where each entry is the mean intensity of the gradient
# over a specific feature-map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))
print(pooled_grads)
