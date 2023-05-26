import os
import cv2 as cv
import numpy as np
from exposure_fusion import ExposureFusion

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
  
def main():
    #TODO: Add path for the images
    image_float32 = open_images('./images')
    # The two modes are 'pyramids' or 'naive'
    mode = 'naive'
    fusion = ExposureFusion(mode)
    hdr = fusion(image_float32)
    cv.imshow(f"Final HDR image, {mode.upper()}", hdr)
    cv.imwrite(f"./out/{mode.upper()}.jpg", hdr)

if __name__ == "__main__":
    main()
    cv.waitKey(0)