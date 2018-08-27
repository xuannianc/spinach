import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

inputs = Input(shape=(258, 540, 3))
x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
encoded = MaxPooling2D(2, 2)(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
autoencoder = Model(input=inputs, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
