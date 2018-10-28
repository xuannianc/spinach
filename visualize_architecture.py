# import the necessary packages
from nn.cnn.lenet import LeNet
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16

# initialize LeNet and then write the network architecture
# visualization graph to disk
# model = LeNet.build(28, 28, 1, 10)
# plot_model(model, to_file="lenet.jpg", show_shapes=True)

vgg = VGG16(weights='imagenet')
plot_model(vgg, to_file='vgg16.jpg', show_shapes=True)
