from typing import Any
import cv2 as cv
import numpy as np

from dataclasses import dataclass
@dataclass
class Exponents():
    contr: float
    sat: float
    exp: float

class ExposureFusion():
    def __init__(self, mode: str, exp: Exponents = Exponents(1.0, 1.0, 1.0), sigma: float = 0.2, pyramid_layers: int = 4):
        # Parameters needed for the exposure fusion
        self.mode: str = mode
        self.sigma: float = sigma
        self.pyramid_layers: int = pyramid_layers
        self.exponents: Exponents = exp

        assert pyramid_layers >= 1, "Pyramid levels must be at least 1."
        assert sigma > 0, "Sigma must be positive."
    
    
    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Description
        """
        try:
            assert len(images) >= 3, "Not enough images as input."
            assert all([image.shape == images[0].shape for image in images]), "Images in input have different shape."
            assert all([image.shape[-1] == 3 for image in images]), "Images in input must be 3-channeled."
        except AssertionError as e:
            print(f"Invalid input: {e}")
        
        # Copy original images
        images = [image.copy() for image in images] #TODO: Check if really necessary
        
        try:
            weights = self.calc_weights(images)
            
            if self.mode == "pyramids":
                pyramids_gauss, pyramids_lap =  self.build_pyramids(images, weights)
                pyramid_final = self.blend_pyramids(pyramids_gauss, pyramids_lap, len(images))
                final_image = self.collapse_pyramid(pyramid_final)
            elif self.mode == "naive":
                final_image = self.blend_naive(images, weights)
        except:
            raise
        
        return final_image
           
            
    # Calculates the 3 weights for all the images
    def calc_weights(self, images: list[np.ndarray]):
        weight_sum = np.zeros(images[0].shape[:2], dtype=np.float32)
        weights = []
        
        for image in images:
            # W = np.ones(image.shape[:2], dtype=np.float32)
            w_contrast = self.calc_weights_contrast(image)
            w_saturation = self.calc_weights_saturation(image)
            w_exposure = self.calc_weights_exposure(image)
            
            image_weight = ((w_contrast ** self.exponents.contr) + 1) \
                        * ((w_saturation ** self.exponents.sat) + 1) \
                        * ((w_exposure ** self.exponents.exp) + 1)
            weights.append(image_weight)
            weight_sum += image_weight
            
        weights = [np.uint8((w / weight_sum)*255) for w in weights]
        
        return weights
    
    
    # Calculates the contrast metric used for the creation of the weight map
    def calc_weights_contrast(self, image: np.ndarray):
        image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image_laplacian = cv.Laplacian(image_gray, ddepth=-1)   #TODO: Check why ddepth = -1
        w_contrast = np.absolute(image_laplacian)
        
        return w_contrast
    
    
    # Calculates the saturation metric used for the creation of the weight map
    def calc_weights_saturation(self, image: np.ndarray):
        w_saturation = image.std(axis=2, dtype=np.float32)
        
        return w_saturation
    
    
    # Calculates the exposure metric used for the creation of the weight map
    def calc_weights_exposure(self, image: np.ndarray):
        w_exposure = np.prod(np.exp(-((image - 0.5)**2) / (2*self.sigma)), axis = 2, dtype=np.float32)
        
        return w_exposure
    
    
    def blend_naive(self, images: list[np.ndarray], weights: list[np.ndarray]):
        final_image = np.zeros(images[0].shape, dtype=np.float32)
        
        for image in range(len(images)):
            final_image[...,0] += weights[image] * images[image][...,0]
            final_image[...,1] += weights[image] * images[image][...,1]
            final_image[...,2] += weights[image] * images[image][...,2]
            
        return final_image.astype(np.uint8)
    
    
    def build_pyramids(self, images: list[np.ndarray], weights: list[np.ndarray]):
        pyramids_gauss = []
        pyramids_lap = []
        
        for i in range(len(images)):
            pyramids_gauss.append(self.build_pyramid_gauss(weights[i]))
            pyramids_lap.append(self.build_pyramid_lap(self.build_pyramid_gauss(images[i])))
        
        
        return pyramids_gauss, pyramids_lap
    
    
    def build_pyramid_gauss(self, image) -> list[np.ndarray]:
        pyramid_gauss = []
        
        low_image = image.copy()
        pyramid_gauss.append(low_image)
        
        for _ in range(self.pyramid_layers):
            low_image = cv.pyrDown(low_image)
            pyramid_gauss.append(low_image)
        
        return pyramid_gauss
        
    
    def build_pyramid_lap(self, pyramid_gauss) -> list[np.ndarray]:
        pyramid_lap = []
        
        pyramid_lap.append(pyramid_gauss[-1])
        
        for i in range(self.pyramid_layers, 0, -1):
            size = (pyramid_gauss[i-1].shape[1], pyramid_gauss[i-1].shape[0])
            up_image = cv.pyrUp(pyramid_gauss[i], dstsize=size)
            laplacian_image = cv.subtract(pyramid_gauss[i-1], up_image)
            pyramid_lap.append(laplacian_image)
        
        pyramid_lap.reverse()
        
        return pyramid_lap
    
    
    def blend_pyramids(self, pyramids_gauss, pyramids_lap, number_images) -> list[np.ndarray]:
        pyramid_final = []
        for layer in range(self.pyramid_layers + 1):
            layer_sum = np.zeros(pyramids_lap[0][layer].shape, dtype=np.float32)
            for image in range(number_images):
                layer_sum[...,0] += pyramids_lap[image][layer][...,0] * pyramids_gauss[image][layer]
                layer_sum[...,1] += pyramids_lap[image][layer][...,1] * pyramids_gauss[image][layer]
                layer_sum[...,2] += pyramids_lap[image][layer][...,2] * pyramids_gauss[image][layer]
            
            pyramid_final.append(layer_sum)
            
        return pyramid_final
    
    
    def collapse_pyramid(self, pyramid_final) -> np.ndarray:
        result_image = pyramid_final[-1].copy()
        
        for layer in range(self.pyramid_layers, 0, -1):
            size = (pyramid_final[layer-1].shape[1], pyramid_final[layer-1].shape[0])
            up_image = cv.pyrUp(result_image, dstsize=size)
            result_image = cv.add(up_image, pyramid_final[layer-1], dtype=cv.CV_8UC3)
        
        return result_image