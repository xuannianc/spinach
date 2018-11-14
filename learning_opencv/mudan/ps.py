import cv2
import numpy as np
import imutils


def move_logo_to_center():
    pass


logo_image = cv2.imread('pwc_logo.jpg')
height, width = logo_image.shape[:2]
new_height = height // 8
new_width = width // 8
print(new_height, new_width)
resized_logo_image = cv2.resize(logo_image, (new_width, new_height))
cv2.imshow('resized_logo_image', resized_logo_image)
cv2.waitKey(0)
mudan_image = cv2.imread('mudan.jpg')
center_y = 505
center_x = 372
cv2.circle(mudan_image, (center_x, center_y), 60, (255, 255, 255), -1)
cv2.imshow('mudan', mudan_image)
cv2.waitKey(0)
start_y = center_y - new_height // 2
end_y = center_y + (new_height - new_height // 2)
start_x = center_x - new_width // 2
end_x = center_x + (new_width - new_width // 2)
mudan_image[start_y + 8:end_y - 8, start_x + 8:end_x - 8] = resized_logo_image[8:-8, 8:-8]
cv2.imshow('mudan', mudan_image)
cv2.waitKey(0)
############## pr ###############
# pr_logo_image = cv2.imread('pr_logo.jpg')
# h, w = pr_logo_image.shape[:2]
# resized_pr_logo_image = cv2.resize(pr_logo_image, (w // 4, h // 4))
# pr_h,pr_w = resized_pr_logo_image.shape[:2]
# cv2.imshow('resized_pr_logo_image', resized_pr_logo_image)
# cv2.waitKey(0)
# resized_pr_logo_image[np.where((resized_pr_logo_image == [255, 255, 255]).all(axis=2))] = [197,232,245]
# cv2.imshow('resized_pr_logo_image', resized_pr_logo_image)
# cv2.waitKey(0)
# pr_center_x = 220
# pr_center_y = 300
# start_y = pr_center_y - pr_h // 2
# end_y = pr_center_y + (pr_h - pr_h // 2)
# start_x = pr_center_x - pr_w // 2
# end_x = pr_center_x + (pr_w - pr_w // 2)
# mudan_image[start_y + 10:end_y, start_x:end_x] = resized_pr_logo_image[10:]
# cv2.imshow('mudan', mudan_image)
# cv2.waitKey(0)

def attach_logo(logo_image, resized_height, center_x, center_y, crop=None, colorize=True):
    resized_logo_image = imutils.resize(logo_image, height=resized_height)
    resized_h, resized_w = resized_logo_image.shape[:2]
    cv2.imshow('resized_logo_image', resized_logo_image)
    cv2.waitKey(0)
    start_y = center_y - resized_h // 2
    end_y = center_y + (resized_h - resized_h // 2)
    start_x = center_x - resized_w // 2
    end_x = center_x + (resized_w - resized_w // 2)
    if colorize:
        lower_white = np.array([220, 220, 220], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(resized_logo_image, lower_white, upper_white)
        resized_logo_image[np.where((mask == 255))] = [197, 232, 245]
    cv2.imshow('colored_resized_logo_image', resized_logo_image)
    cv2.waitKey(0)
    if crop:
        mudan_image[start_y + crop[0]:end_y - crop[1], start_x + crop[2]:end_x - crop[3]] = resized_logo_image[
                                                                                            crop[0]:resized_h - crop[1],
                                                                                            crop[2]:resized_w - crop[3]]
    else:
        mudan_image[start_y:end_y, start_x:end_x] = resized_logo_image
    cv2.imshow('mudan', mudan_image)
    cv2.waitKey(0)


pr_logo_image = cv2.imread('pr_logo.jpg')
attach_logo(pr_logo_image, resized_height=100, center_x=220, center_y=300, crop=(10, 10, 0, 0))
gsk_logo_image = cv2.imread('gsk_logo.jpeg')
attach_logo(gsk_logo_image, resized_height=80, center_x=516, center_y=297)
jiaji_logo_image = cv2.imread('jiaji_logo.jpg')
attach_logo(jiaji_logo_image, resized_height=100, center_x=100, center_y=370, crop=(0, 5, 0, 0))
dhl_logo_image = cv2.imread('dhl_logo.png')
attach_logo(dhl_logo_image, resized_height=100, center_x=630, center_y=360, crop=(10, 20, 0, 0))
cv2.imwrite('mudan_pwc.jpg', mudan_image)
