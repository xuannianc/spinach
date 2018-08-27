# Module with image-preprocessing utilities
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os
import os.path as osp
from dogs_vs_cats.config import DATASET_DIR
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_cats_dir = osp.join(DATASET_DIR, 'data', 'train', 'cat')
image_paths = [osp.join(train_cats_dir, image_file) for image_file in os.listdir(train_cats_dir)]
image_path = image_paths[3]
# Chooses one image to augment, return a PIL Image
image = load_img(image_path, target_size=(150, 150))
print(type(image))
# Converts it to a Numpy arra
x = img_to_array(image)
print(type(x))
# Reshapes it to (1, 150, 150, 3)
i = 0
x = x.reshape((1,) + x.shape)
# Generates batches of randomly transformed images. Loops indefinitely,
# so you need to break the loop at some point!
for batch in datagen.flow(x, batch_size=1):
    print('batch={}'.format(batch))
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
