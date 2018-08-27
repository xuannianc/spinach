import os
import os.path as osp

DATASET_DIR = '/home/adam/.keras/datasets/kaggle_denoising_dirty_documents'
TRAIN_DIR = osp.join(DATASET_DIR, 'train')
LABEL_DIR = osp.join(DATASET_DIR, 'train_cleaned')
TEST_DIR = osp.join(DATASET_DIR, 'test')
