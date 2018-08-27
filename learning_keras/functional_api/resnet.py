from keras import layers
x = ...
# Applies a transformation to x
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
# Adds the original x back to the output features
y = layers.add([y, x])

from keras import layers
x = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)
# Uses a 1 × 1 convolution to linearly downsample the original tensor x to the same shape as y
residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
# Adds the residual tensor back to the output features
y = layers.add([y, residual])
