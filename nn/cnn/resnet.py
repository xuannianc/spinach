# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K


class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chan_dim, reduce=False,
                        regularization=0.0001, bn_epsilon=2e-5, bn_momentum=0.9):
        """
        残缺模块
        :param data: 输入
        :param K: 最后一个 conv 的 kernel 个数, 前两个 conv 的 kernel 个数就是 K/4
        :param stride: conv 的 stride
        :param chan_dim: channel 的维度序号, tf 就是 -1
        :param reduce: 是否减少维度
        :param regularization: 不太懂
        :param bn_epsilon: 不太懂,防止除 0
        :param bn_momentum: 不太懂
        :return:
        """
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data
        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_epsilon,
                                 momentum=bn_momentum)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(regularization))(act1)
        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_epsilon,
                                 momentum=bn_momentum)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
                       padding="same", use_bias=False,
                       kernel_regularizer=l2(regularization))(act2)
        # the third block of the ResNet module is another set of 1x1
        # CONVs
        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_epsilon,
                                 momentum=bn_momentum)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False,
                       kernel_regularizer=l2(regularization))(act3)
        # if we are to reduce the spatial size, apply a CONV layer to
        # the shortcut
        if reduce:
            shortcut = Conv2D(K, (1, 1), strides=stride,
                              use_bias=False, kernel_regularizer=l2(regularization))(act1)
        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(height, width, depth, num_classes, stages, filters,
              regularization=0.0001, bn_epsilon=2e-5, bn_momentum=0.9, chan_dim=-1, dataset="cifar"):
        """
        构建 resnet
        :param height:
        :param width:
        :param depth:
        :param num_classes:
        :param stages: list,每一个 stage/stack 的残差模块的个数,如 [3,4,6] 表示第一个 stack 有 3 个残差模块
        :param filters: list,每一个 stage/stack filter 的个数
        :param regulariztion: rela
        :param bn_epsilon:
        :param bn_momentum:
        :param chan_dim:
        :param dataset:
        :return:
        """
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        input_shape = (height, width, depth)
        # set the input and apply BN
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim, epsilon=bn_epsilon, momentum=bn_momentum)(inputs)
        # check if we are utilizing the CIFAR dataset
        if dataset == "cifar":
            # apply a single CONV layer
            x = Conv2D(filters[0], (3, 3), use_bias=False,
                       padding="same", kernel_regularizer=l2(regularization))(x)
        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            # stride = (1,1) 表示不改变图像的 size
            # 第一个 stage 不改变图像的 size, 后面的 stage 改变
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                                       chan_dim, reduce=True, bn_epsilon=bn_epsilon,
                                       bn_momentum=bn_momentum)
            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1), chan_dim, bn_epsilon=bn_epsilon,
                                           bn_momentum=bn_momentum)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chan_dim, epsilon=bn_epsilon,
                               momentum=bn_momentum)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        # softmax classifier
        x = Flatten()(x)
        x = Dense(num_classes, kernel_regularizer=l2(regularization))(x)
        x = Activation("softmax")(x)
        # create the model
        model = Model(inputs, x, name="resnet")
        # return the constructed network architecture
        return model
