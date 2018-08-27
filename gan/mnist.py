from keras.layers import Dense, Activation, BatchNormalization, Reshape, Conv2D, UpSampling2D
from keras.layers import MaxPooling2D, Flatten, Input, LeakyReLU, Dropout
from keras.models import Sequential, Model
import numpy as np
from keras.datasets import mnist


def generator_model():
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch * 14 * 14, kernel_initializer='glorot_normal')(g_input)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Reshape((14, 14, nch))(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(int(nch / 2), (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Conv2D(int(nch / 4), (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)


generator = generator_model()
generator.summary()


def discriminator_model():
    d_input = Input((28, 28, 1))
    nch = 512
    H = Conv2D(int(nch / 2), (5, 5), padding='same',
               activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.5)(H)
    H = Conv2D(nch, (5, 5), padding='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.5)(H)
    H = Flatten()(H)
    H = Dense(int(nch / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.5)(H)
    d_V = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_V)


discriminator = discriminator_model()
discriminator.summary()

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
x_train = xtrain.astype(np.float32) / 255.0
x_test = xtrain.astype(np.float32) / 255.0
if __name__ == "__main__":
    # z in R^100
    latent_dim = 100
    # x in R^{28x28}
    input_shape = (28, 28, 1)
    # generator (z -> x)
    generator = generator_model()
    # discriminator (x -> y)
    discriminator = discriminator_model()
    # gan (x - > yfake, yreal), z generated on GPU
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))
    # print summary of models
    generator.summary()
    discriminator.summary()
    gan.summary()
