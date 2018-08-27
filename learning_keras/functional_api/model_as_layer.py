from keras import layers
from keras import applications
from keras import Input
from keras.models import Model

# The base image-processing model is the Xception network (convolutional base only).
xception_base = applications.Xception(weights=None,
                                      include_top=False)
# The inputs are 250 × 250 RGB images.
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))
# Calls the same vision model twice
# (None, 8, 8, 2048)
left_features = xception_base(left_input)
print(left_features.shape)
right_features = xception_base(right_input)
# The merged features contain information from the right visual feed and the left visual feed.
# (None, 8, 8, 4096)
merged_features = layers.concatenate([left_features, right_features], axis=-1)
model = Model([left_input, right_input], merged_features)
model.summary()
