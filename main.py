import os
import argparse
import cv2 as cv
import numpy as np
from exposure_fusion import ExposureFusion

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str)
args = parser.parse_args()
mode = args.mode

def open_images(images_dir: str = './images') -> list[np.ndarray]:
    images = []
    
    # Checks on the input
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
            image = np.float32(cv.imread(images_dir + '/' +file)) / 255.0
            images.append(image)

    except NotADirectoryError:
        print(f"Input Error: Directory '{images_dir}' doesn't exist.")
    except FileNotFoundError as e:
        print(f"Input Error: {e}")
        
    return images
  
def main(mode):
    #TODO: Add path for the images
    image_float32 = open_images('./images')
    fusion = ExposureFusion(mode)
    try:
        if mode=="pyramid":
            hdr, canvas = fusion(image_float32)
            cv.imshow(f"Final HDR image, {mode.upper()}", hdr)
            cv.imshow(f"Laplacan Pyramids", canvas)
            cv.imwrite(f"./out/{mode.upper()}.jpg", hdr, [cv.IMWRITE_JPEG_QUALITY, 100])    #TODO: Create better lables for the files
            cv.imwrite(f"./out/{mode.upper()}_pyramid.jpg", canvas, [cv.IMWRITE_JPEG_QUALITY, 100])
        elif mode == "naive":
            hdr = fusion(image_float32)
            cv.imshow(f"Final HDR image, {mode.upper()}", hdr)
            cv.imwrite(f"./out/{mode.upper()}.jpg", hdr, [cv.IMWRITE_JPEG_QUALITY, 100])
        else:
            raise ValueError
    except ValueError as e:
        print(f"{type(e).__name__}: '{mode}' mode doesn't exist. Try between 'naive' or 'pyramids'")
    


if __name__ == "__main__":
    main(mode)
    cv.waitKey(0)