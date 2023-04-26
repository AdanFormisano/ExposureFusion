import cv2 as cv
import numpy as np
from scipy import stats
import os

dictImg = dict()



def Constrast(img):
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    laplacian = cv.Laplacian(img_gray, cv.CV_32F)
    C = cv.convertScaleAbs(laplacian)

    return C

# TODO: sposta sigma dentro a exponential_euclidean
def SaturationExposure(img, C, Wsum):
    sigma = 0.2
    result = list()
    
    for Y in range(len(img)):
        result.append([])
        for X in range(len(img[0])):
            S = stats.tstd(img[Y][X])
            B, R, G = img[Y][X][0], img[Y][X][1], img[Y][X][2]
            red_exp = exponential_euclidean(R, sigma)
            green_exp = exponential_euclidean(G, sigma)
            blue_exp = exponential_euclidean(B, sigma)
            E = red_exp * green_exp * blue_exp


            prod = E * S * C[Y][X] + 1e-12
            result[Y].append(prod)
            Wsum[Y][X] += prod
    
    return result

def exponential_euclidean(canal, sigma):
    return np.exp(-(canal - 0.5)**2 / 0.08)

def BuildW():
    #  W = {mean:[[C * S * E][]], under:[[C * S * E][]], over:[[C * S * E]]}
    W = dict()
    O = dict()
    sizeImg = cv.imread('images/' + os.listdir('images')[0])
    Wsum = np.zeros((sizeImg.shape[0], sizeImg.shape[1]))
    R = [[0]*sizeImg.shape[1]]*sizeImg.shape[0] # ???

    for fileImg in os.listdir('images'):
        path = 'images/' + fileImg
        imgName = fileImg[:-4]
        img = cv.imread(path)

        O[imgName] = img

        # Inserisce una lista per ogni foto
        W[imgName] = SaturationExposure(img, Constrast(img), Wsum)
    
    for Y in range(sizeImg.shape[0]):
        for X in range(sizeImg.shape[1]):
            r = [0,0,0]
            for img in O:
                I = O[img][Y][X]
                Wcappello = W[img][Y][X]/Wsum[Y][X]
                # print(type(Wcappello), type(I[0]), type(img))
                r[0] += Wcappello * I[0]
                r[1] += Wcappello * I[1]
                r[2] += Wcappello * I[2]

            R[Y][X] = tuple(r)
    print(R[123][123])

# cv.imshow('R', R)
# cv.imshow('Mean', img_mean)
# cv.imshow('Over', img_over)
# cv.imshow('Under', img_under)

# np.seterr(all='print')

BuildW()
