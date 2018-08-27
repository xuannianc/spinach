from keras.applications import resnet50

resnet = resnet50.ResNet50(input_shape=(224,224,3),
                           include_top=True,
                           weights='imagenet',
                           pooling='max')
resnet.summary()
