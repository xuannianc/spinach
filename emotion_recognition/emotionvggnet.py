# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, num_classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
        # Block #1: first CONV => RELU => CONV => RELU => POOL
        # layer set
        model.add(Conv2D(32, (3, 3), padding="same",
                         kernel_initializer="he_normal", input_shape=input_shape))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Block #2: second CONV => RELU => CONV => RELU => POOL
        # layer set
        model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Block #3: third CONV => RELU => CONV => RELU => POOL
        # layer set
        model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Block #6: second set of FC => RELU layers
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Block #7: softmax classifier
        model.add(Dense(num_classes, kernel_initializer="he_normal"))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model
