import shutil
import os.path as osp
import os
from dogs_vs_cats.config import DATASET_DIR


def copy_files(prefix_str, range_start, range_end, target_dir):
    image_paths = [osp.join(DATASET_DIR, 'train', prefix_str + '.' + str(i) + '.jpg')
                   for i in range(range_start, range_end)]
    dest_dir = osp.join(DATASET_DIR, 'data', target_dir, prefix_str)
    if not osp.exists(dest_dir):
        os.makedirs(dest_dir)
    for image_path in image_paths:
        shutil.copy(image_path, dest_dir)


copy_files('dog', 0, 1000, 'train')
copy_files('cat', 0, 1000, 'train')
copy_files('dog', 1000, 1500, 'val')
copy_files('cat', 1000, 1500, 'val')
copy_files('dog', 1500, 2000, 'test')
copy_files('cat', 1500, 2000, 'test')
