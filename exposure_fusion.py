import sys
import os

import cv2 as cv
import numpy as np

from dataclasses import dataclass


def open_images(images_dir: str) -> tuple[int, list[np.ndarray]]:
    images = []
    n_images = 0

    # Runs checks on the input
    try:
        # Check if the path exists
        if not os.path.exists(images_dir):
            raise ValueError

        # Checks if the directory is empty
        elif len(os.listdir(images_dir)) == 0:
            raise FileNotFoundError(f"Directory {images_dir} is empty")

        else:
            # Checks if the files are .jpg
            for file in os.listdir(images_dir):
                if os.path.splitext(file)[1] != '.jpg':
                    raise FileNotFoundError(f"File {file} is not a jpg")

                # Read image and convert to float32 for processing
                image = np.float32(cv.imread(f"{images_dir}/{file}")) / 255.0
                images.append(image)
                n_images += 1

            return n_images, images

    except Exception:
        raise


@dataclass
class Exponents:
    contr: float
    sat: float
    exp: float


class ExposureFusion:
    def __init__(self, mode: str, n_images: int, exp: Exponents = Exponents(1.0, 1.0, 1.0), sigma: float = 0.2,
                 pyramid_layers: int = 4):
        # Parameters needed for the exposure fusion
        self.mode: str = mode
        self.n_images = n_images
        self.sigma: float = sigma
        self.pyramid_layers: int = pyramid_layers
        self.exponents: Exponents = exp

    def __call__(self, images_original: list[np.ndarray]):
        # Checks the input images
        try:
            assert self.n_images >= 3, "Not enough images as input."
            assert all([image.shape == images_original[0].shape for image in
                        images_original]), "Images in input have different shape."
            assert all([image.shape[-1] == 3 for image in images_original]), "Images in input must be 3-channeled."
        except AssertionError as e:
            print(f"Invalid input: {e}")

        # Copy original images
        images = [image.copy() for image in images_original]    # CHECK: is it needed to do a copy?

        try:
            weights = self.calc_weights(images)

            if self.mode == "pyramids":
                # Calculates the pyramids need for the multi-resolution fusion
                pyramids_gauss, pyramids_lap = self.build_pyramids(images, weights)
                # Build the final laplacian pyramid
                pyramid_final = self.blend_pyramids(pyramids_gauss, pyramids_lap)
                # Collapses the final pyramid to create the final hdr image
                final_image = self.collapse_pyramid(pyramid_final)

                # Creates an image with all the pyramid layers represented in it
                # canvas_pyramid = self.make_canvas(pyramids_gauss[1])

                return final_image

            elif self.mode == "naive":
                # Creates the final image using the naive method for the fusion
                final_image = self.blend_naive(images, weights)
                return final_image

        except Exception:
            raise

    def calc_weights(self, images: list[np.ndarray]):
        """Calculates the weights for all the images."""

        weight_sum = np.zeros(images[0].shape[:2], dtype=np.float32)
        weights = []

        for image in images:
            w_contrast = self.calc_weights_contrast(image)
            w_saturation = self.calc_weights_saturation(image)
            w_exposure = self.calc_weights_exposure(image)

            image_weight = ((w_contrast ** self.exponents.contr) + 1) *\
                           ((w_saturation ** self.exponents.sat) + 1) *\
                           ((w_exposure ** self.exponents.exp) + 1)

            weights.append(image_weight)
            weight_sum += image_weight

        # Normalizing the weights values
        weights = [np.uint8((w / weight_sum) * 255) for w in weights]

        return weights

    def calc_weights_contrast(self, image: np.ndarray) -> np.ndarray:
        """Calculates the contrast metric used for the creation of the weight map."""
        image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image_laplacian = cv.Laplacian(image_gray, ddepth=-1)  # TODO: Check why ddepth = -1
        w_contrast = np.absolute(image_laplacian)

        return w_contrast

    def calc_weights_saturation(self, image: np.ndarray):
        """Calculates the saturation metric used for the creation of the weight map"""
        w_saturation = image.std(axis=2, dtype=np.float32)

        return w_saturation

    def calc_weights_exposure(self, image: np.ndarray):
        """Calculates the exposure metric used for the creation of the weight map"""
        w_exposure = np.prod(np.exp(-((image - 0.5) ** 2) / (2 * self.sigma)), axis=2, dtype=np.float32)

        return w_exposure

    def blend_naive(self, images: list[np.ndarray], weights: list[np.ndarray]):
        """Naive method for exposure blending."""
        final_image = np.zeros(images[0].shape, dtype=np.float32)

        for image in range(self.n_images):
            final_image[..., 0] += weights[image] * images[image][..., 0]
            final_image[..., 1] += weights[image] * images[image][..., 1]
            final_image[..., 2] += weights[image] * images[image][..., 2]

        return final_image.astype(np.uint8)

    def build_pyramids(self, images: list[np.ndarray], weights: list[np.ndarray]):
        """Builds the pyramids needed for the multi-resolution fusion method."""
        pyramids_gauss = []
        pyramids_lap = []

        for i in range(self.n_images):
            pyramids_gauss.append(self.build_pyramid_gauss(weights[i]))
            pyramids_lap.append(self.build_pyramid_lap(self.build_pyramid_gauss(images[i])))

        return pyramids_gauss, pyramids_lap

    def build_pyramid_gauss(self, image) -> list[np.ndarray]:
        """Builds the gaussian pyramid."""
        pyramid_gauss = []

        low_image = image.copy()
        pyramid_gauss.append(low_image)

        for _ in range(self.pyramid_layers):
            low_image = cv.pyrDown(low_image)
            pyramid_gauss.append(low_image)

        return pyramid_gauss

    def build_pyramid_lap(self, pyramid_gauss) -> list[np.ndarray]:
        """Builds the laplacian pyramid."""
        pyramid_lap = [pyramid_gauss[-1]]

        for i in range(self.pyramid_layers, 0, -1):
            size = (pyramid_gauss[i - 1].shape[1], pyramid_gauss[i - 1].shape[0])

            up_image = cv.pyrUp(pyramid_gauss[i], dstsize=size)
            laplacian_image = cv.subtract(pyramid_gauss[i - 1], up_image)
            pyramid_lap.append(laplacian_image)

        pyramid_lap.reverse()

        return pyramid_lap

    def blend_pyramids(self, pyramids_gauss, pyramids_lap) -> list[np.ndarray]:
        """Blends the gaussian and laplacian pyramids to create a new laplacian pyramid that is then used to create
        the final image."""
        pyramid_final = []
        for layer in range(self.pyramid_layers + 1):
            layer_sum = np.zeros(pyramids_lap[0][layer].shape, dtype=np.float32)
            for image in range(self.n_images):
                layer_sum[..., 0] += pyramids_lap[image][layer][..., 0] * pyramids_gauss[image][layer]
                layer_sum[..., 1] += pyramids_lap[image][layer][..., 1] * pyramids_gauss[image][layer]
                layer_sum[..., 2] += pyramids_lap[image][layer][..., 2] * pyramids_gauss[image][layer]

            pyramid_final.append(layer_sum)

        return pyramid_final

    def collapse_pyramid(self, pyramid_final) -> np.ndarray:
        """Builds the final hdr image collapsing the final pyramid."""
        result_image = pyramid_final[-1].copy()

        for layer in range(self.pyramid_layers, 0, -1):
            size = (pyramid_final[layer - 1].shape[1], pyramid_final[layer - 1].shape[0])
            up_image = cv.pyrUp(result_image, dstsize=size)
            result_image = cv.add(up_image, pyramid_final[layer - 1], dtype=cv.CV_8UC3)

        return result_image

    def make_canvas(self, images: list[np.ndarray]):
        """Builds an image where all the layers of a pyramid are represented in the same canvas one next to the
        other."""
        max_height = images[0].shape[0]

        canvas = images[0].copy()

        current_x = 0
        for img in images[1:]:
            canvas[max_height - img.shape[0]:, current_x:current_x + img.shape[1]] = img
            current_x += img.shape[1]

        return canvas
