import cv2 as cv
import numpy as np
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
    image = np.float32(image_uint)/255.0
    og_Img.append(image)
    W = np.ones(image.shape[:2], dtype=np.float32)
    
    # Contrast
    img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    laplacian = cv.Laplacian(img_gray, ddepth=-1)
    W_contrast = np.absolute(laplacian) ** w_c + 1              # + 1 per renderlo "decente"
    # C = cv.convertScaleAbs(laplacian)
    W = np.multiply(W,W_contrast)

    # saturation
    W_saturation = image.std(axis=2, dtype=np.float32) ** w_s + 1               # + 1 per renderlo "decente"
    W = np.multiply(W, W_saturation)

    # well-exposedness
    sigma = 0.2
    W_exposedness = np.prod(np.exp(-((image - 0.5)**2)/(2*sigma)), axis=2, dtype=np.float32) ** w_e + 1             # + 1 per renderlo "decente"
    W = np.multiply(W, W_exposedness)

    weights_sum += W
    weights.append(W)

# Naive implementation
def Naive(weights, weights_sum, og_Img):
    for i in range(len(weights)):
        # weights e' la lista contentente i pesi delle singole foto aka "W cappello"
        weights[i] /= weights_sum
        weights[i] = np.uint8(weights[i]*255)

        b = og_Img[i][...,0]
        g = og_Img[i][...,1]
        r = og_Img[i][...,2]

        R[...,0] += weights[i] * b
        R[...,1] += weights[i] * g
        R[...,2] += weights[i] * r

    R = R.astype(dtype=np.uint8)
    cv.imshow('R',R)

# Pyramid implementation
def Pyramid(images, weights, weights_sum, show:bool):    # RICORDA LA VARIABILE SHOW!!!!!
    ImgLP = []
    WeiGP = []
    LFinal = []

    for img in range(len(images)):
        weights[img] /= weights_sum
        weights[img] = np.uint8(weights[img]*255)
        # Laplacian pyramid of all images
        ImgLP.append(LPyr(GPyr(images[img], False), images[img]))
        WeiGP.append(GPyr(weights[img], True))    #DEBUG: Crea una foto in piu' rispetto a LPyr
    

    # Loop su ogni livello della piramide
    for lvl in range(5):
        lvlSum = np.zeros(ImgLP[0][lvl].shape, dtype=np.float32)
        # Loop su ogni image
        for img in range(len(images)):
            lvlSum[...,0] += ImgLP[img][lvl][...,0] * WeiGP[img][lvl]
            lvlSum[...,1] += ImgLP[img][lvl][...,1] * WeiGP[img][lvl]
            lvlSum[...,2] += ImgLP[img][lvl][...,2] * WeiGP[img][lvl]

        LFinal.append(lvlSum)

    R = LFinal[-1].copy()
    # R = np.uint8(R)
    for Llvl in range(4, 0, -1):
        if Llvl == 4:
            cv.imshow("PRE R", R)
        size = (LFinal[Llvl-1].shape[1], LFinal[Llvl-1].shape[0])
        up = cv.pyrUp(R, dstsize=size)
        R = cv.add(up, LFinal[Llvl-1], dtype=cv.CV_8UC3)

    # Show Pyramid
    if show:
        # imageNumber = 0
        # for i in range(len(ImgLP)):
        #     windowNumber = 0
        #     for img in ImgLP[i]:
        #         windowName = str(imageNumber) +  " " + str(windowNumber) + " Laplacian"
        #         cv.imshow(windowName, img)
        #         windowNumber += 1

        #     windowNumber = 0
        #     for img in WeiGP[i]:
        #         windowName = str(imageNumber) + " " + str(windowNumber) + " Gaussian"
        #         cv.imshow(windowName, img)
        #         windowNumber += 1

        #     imageNumber += 1
        label = 0
        for i in range(len(LFinal)):
            cv.imshow(str(label), LFinal[i])
            label += 1
        cv.imshow("R",R)

# Builds Gaussian pyramid
def GPyr(img, W:bool):
    gPyr = []
    #if W:
    #    gPyr.append(img)
    lowImg = img.copy()
    gPyr.append(lowImg)
    for _ in range(4):
        lowImg = cv.pyrDown(lowImg)
        gPyr.append(lowImg)
    return gPyr

#TODO: Controllare che la Laplacyan pyramid finale sia corretta.
# Builds Laplacian pyramid
def LPyr(gPyr,img):
    lPyr = []
    lPyr.append(gPyr[-1])
    for i in range(4,0,-1):
        size = (gPyr[i-1].shape[1], gPyr[i-1].shape[0])
        GP = cv.pyrUp(gPyr[i], dstsize=size)
        L = cv.subtract(gPyr[i-1], GP)
        lPyr.append(L)
    # lPyr.append(cv.Laplacian(img, ddepth=-1))
    lPyr.reverse()
    return lPyr

#cv.imshow('L', cv.Laplacian(R, ddepth=-1))
# cv.imshow('W', weights[2])
# cv.imshow('lowR',lowR)
Pyramid(og_Img, weights, weights_sum, True)             # RICORDA LA VARIABILE SHOW!!!!!
cv.waitKey(0)

# W = weights(images, img_demo, R, weights_sum, weights)
