from keras import layers
import numpy as np
from keras.models import Model

# x = np.random.randint(0, 255, (100, 299, 299, 3))
x = layers.Input((298, 298, 3))
# Every branch has the same stride value (2),
# which is necessary to keep all branch outputs
# the same size so you can concatenate them.
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
# In this branch, the striding occurs
# in the spatial convolution layer.
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
# In this branch, the striding occurs
# in the average pooling layer.
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
# Concatenates the branch outputs to obtain the module output
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
model = Model(x, output)
model.summary()
