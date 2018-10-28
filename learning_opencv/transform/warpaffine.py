import cv2
import numpy as np


def flipx(image):
    h, w = image.shape[:2]
    matrix = np.array([[-1, 0, 0], [0, 1, 0]]).astype(np.float32)
    image = cv2.warpAffine(image, matrix, dsize=(w, h))
    cv2.imshow('flipx', image)
    cv2.waitKey(0)


def rotate(image):
    h, w = image.shape[:2]
    matrix = np.array([[0, -1, 0], [1, 0, 0]]).astype(np.float32)
    image = cv2.warpAffine(image, matrix, dsize=(w, h))
    cv2.imshow('flipx', image)
    cv2.waitKey(0)


image = cv2.imread('../pikachu.png')
h, w = image.shape[:2]
# M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1)
# image = cv2.warpAffine(image, M, dsize=(w, h))
# flipx(image)
# rotate(image)
pts1 = np.float32([[0, 0], [0, w], [h, 0]])
pts2 = np.float32([[0, w], [0, 0], [h, w]])
matrix = cv2.getAffineTransform(pts1, pts2)
image = cv2.warpAffine(image, matrix, (w, h))
cv2.imshow('image', image)
cv2.waitKey(0)
