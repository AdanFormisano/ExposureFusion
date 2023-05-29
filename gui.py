import tkinter as tk
import os
import cv2 as cv
import numpy as np
from tkinter import filedialog
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from exposure_fusion import ExposureFusion


class TheGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x500")
        self.root.title("Exposure Fusion GUI")
        self.imagefolderpath = tk.StringVar()
        self.implementation = tk.StringVar()

        self.open_button_naive = ttk.Button(
            self.root,
            text='Select folder for Naive Exposure Fusion',
            command=lambda: self.select_folder('naive')
        )
        self.open_button_naive.pack(expand=True, padx=20, pady=10)

        self.open_button_pyramid = ttk.Button(
            self.root,
            text='Select folder for Pyramid Exposure Fusion',
            command=lambda: self.select_folder('pyramids')
        )
        self.open_button_pyramid.pack(expand=True, padx=20, pady=10)

        self.root.mainloop()

    def select_folder(self, implementation):
        foldername = filedialog.askdirectory()  # name of folder that has the images ef will be applied on
        showinfo(
            title='Selected Folder',
            message=foldername
        )
        self.imagefolderpath.set(foldername)
        n_images, image_float32 = open_images(foldername)

        fusion = ExposureFusion(implementation, n_images)
        # Perform exposure fusion based on the selected implementation
        if implementation == 'pyramids':
            hdr, canvas = fusion(image_float32)
            cv.imshow(f"Final HDR image, {implementation.upper()}", hdr)
            cv.imshow(f"Laplacian Pyramids", canvas)
            cv.imwrite(f"./out/{implementation.upper()}.jpg", hdr,
                       [cv.IMWRITE_JPEG_QUALITY, 100])  # TODO: Create better labels for the files
            cv.imwrite(f"./out/{implementation.upper()}_pyramid.jpg", canvas, [cv.IMWRITE_JPEG_QUALITY, 100])
            pass
        elif implementation == 'naive':
            hdr = fusion(image_float32)
            cv.imshow(f"Final HDR image, {implementation.upper()}", hdr)
            cv.imwrite(f"./out/{implementation.upper()}.jpg", hdr, [cv.IMWRITE_JPEG_QUALITY, 100])
            pass


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


myname = TheGUI()
