import os
import argparse
import cv2 as cv
import time

import exposure_fusion
from exposure_fusion import ExposureFusion
import gui

parser = argparse.ArgumentParser()
# group = parser.add_mutually_exclusive_group()
parser.add_argument('-m', '--mode', type=str, choices=["naive", "pyramids"], help="selects which mode to use for the exposure fusion: 'naive' or \
                    'pyramids'")
parser.add_argument('ui', type=str, default="no-gui", choices=["gui", "no-gui"], metavar='mode',
                    help="selects whether to run or not with the GUI")
parser.add_argument('-p', '--path', type=str, help="path to the set of images to elaborate")
args = parser.parse_args()


def run_script():
    fusion_mode = args.mode
    image_path = args.path

    try:
        n_images, image_float32 = exposure_fusion.open_images(image_path)
    except ValueError:
        print(f"Input Error: Directory '{image_path}' doesn't exist.")
    except FileNotFoundError as e:
        print(f"Input Error: {e}")

    else:
        # Starts counter for the elapsed time
        start_time = time.perf_counter()

        fusion = ExposureFusion(fusion_mode, n_images)
        # Runs the exposure fusion
        hdr = fusion(image_float32)

        print(f"--- Done in {round(time.perf_counter() - start_time, 3)} seconds ---")

        # Displays final image
        cv.imshow(f"Final HDR image {fusion_mode.upper()}", hdr)
        cv.imwrite(f"./out/{os.path.split(image_path)[-1]}-{fusion_mode.upper()}.jpg", hdr,
                   [cv.IMWRITE_JPEG_QUALITY, 100])  # TODO: Create better labels for the files


def main():
    # If --no-gui is True run the GUI
    if args.ui == 'gui':
        gui.GUI()
        # TODO: Add warning message if there are arguments
    # Else run directly the script
    else:
        run_script()
        cv.waitKey(0)


if __name__ == "__main__":
    main()
    # cv.waitKey(0)
