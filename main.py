import os
import argparse
import cv2 as cv
import numpy as np
from exposure_fusion import ExposureFusion

parser = argparse.ArgumentParser()
parser.add_argument('run_mode', type=str, help="Selects which mode to use for the exposure fusion: 'naive' or \
                    'pyramids'")
parser.add_argument('--no-gui', action="store_true", help="Runs without the GUI")
args = parser.parse_args()


def open_images(images_dir: str = './images') -> (int, list[np.ndarray]):
    images = []
    n_images = 0

    # Runs checks on the input
    try:
        # Check if the path exists
        if not os.path.exists(images_dir):
            raise NotADirectoryError

        # Checks if the directory is empty
        elif len(os.listdir(images_dir)) == 0:
            raise FileNotFoundError(f"Directory {images_dir} is empty")

        # Checks if the files are .jpg
        for file in os.listdir(images_dir):
            if os.path.splitext(file)[1] != '.jpg':
                raise FileNotFoundError(f"File {file} is not a jpg")

            # Read image and convert to float32 for processing
            image = np.float32(cv.imread(f"{images_dir}/{file}")) / 255.0
            images.append(image)
            n_images += 1

    except NotADirectoryError:
        print(f"Input Error: Directory '{images_dir}' doesn't exist.")
    except FileNotFoundError as e:
        print(f"Input Error: {e}")

    return n_images, images


def main(fusion_mode):
    # TODO: Add path to images
    n_images, image_float32 = open_images('./images')

    try:
        # CHECK: ExposureFusion() in try block?
        fusion = ExposureFusion(fusion_mode, n_images)

        if fusion_mode == "pyramids":
            hdr, canvas = fusion(image_float32)
            cv.imshow(f"Final HDR image, {fusion_mode.upper()}", hdr)
            cv.imshow(f"Laplacian Pyramids", canvas)
            cv.imwrite(f"./out/{fusion_mode.upper()}.jpg", hdr,
                       [cv.IMWRITE_JPEG_QUALITY, 100])  # TODO: Create better labels for the files
            cv.imwrite(f"./out/{fusion_mode.upper()}_pyramid.jpg", canvas, [cv.IMWRITE_JPEG_QUALITY, 100])

        elif fusion_mode == "naive":
            hdr = fusion(image_float32)
            cv.imshow(f"Final HDR image, {fusion_mode.upper()}", hdr)
            cv.imwrite(f"./out/{fusion_mode.upper()}.jpg", hdr, [cv.IMWRITE_JPEG_QUALITY, 100])

        else:
            raise ValueError
    except ValueError as e:
        print(f"{type(e).__name__}: '{fusion_mode}' mode doesn't exist. Try between 'naive' or 'pyramids'")


if __name__ == "__main__":
    main(args.run_mode)
    cv.waitKey(0)
