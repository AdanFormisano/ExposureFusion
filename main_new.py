import cv2 as cv
import numpy as np
from scipy import stats
import os

images = ["images/mask_mean.jpg", "images/mask_over.jpg", "images/mask_under.jpg"]
# images = ["images/HDR_test_scene_1__1.1.1.jpg","images/HDR_test_scene_1__1.1.2.jpg","images/HDR_test_scene_1__1.1.3.jpg","images/HDR_test_scene_1__1.1.4.jpg","images/HDR_test_scene_1__1.1.5.jpg"]
img_demo = cv.imread(images[0])

w_c, w_s , w_e = 1,1,1
weights_sum = np.zeros(img_demo.shape[:2], dtype=np.float32)
R = np.zeros(img_demo.shape, dtype=np.float32)
weights = []

og_Img = []

for image in images:
    image_uint = cv.imread(image)
    image = np.float32(image_uint)/255
    og_Img.append(image)
    W = np.ones(image.shape[:2], dtype=np.float32)
    
    # Contrast
    img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    laplacian = cv.Laplacian(img_gray, cv.CV_32F)
    W_contrast = np.absolute(laplacian) ** w_c
    # C = cv.convertScaleAbs(laplacian)
    W = np.multiply(W,W_contrast)

    # saturation
    W_saturation = image.std(axis=2, dtype=np.float32) ** w_s
    W = np.multiply(W, W_saturation)

    # well-exposedness
    sigma = 0.2
    W_exposedness = np.prod(np.exp(-((image - 0.5)**2)/(2*sigma)), axis=2, dtype=np.float32) ** w_e
    W = np.multiply(W, W_exposedness)

    weights_sum += W
    weights.append(W)

for i in range(len(weights)):
    print("New Img")
    weights[i] /= weights_sum
    weights[i] = np.uint8(weights[i]*255)

    b = og_Img[i][...,0]
    g = og_Img[i][...,1]
    r = og_Img[i][...,2]

    R[...,0] += weights[i] * b
    R[...,1] += weights[i] * g
    R[...,2] += weights[i] * r

cv.imshow('R',R.astype(dtype=np.uint8))
cv.waitKey(0)

# W = weights(images, img_demo, R, weights_sum, weights)