import os
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import numpy as np
import exposure_fusion
from PIL import ImageTk, Image
import cv2 as cv


class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.geometry("1366x768")
        self.root.title("Exposure Fusion")
        self.run_mode_var = StringVar()
        self.image_path = StringVar()
        self.n_images = None
        self.save_path = None

        self.content = ttk.Frame(self.root)
        self.options = ttk.Frame(self.content, borderwidth=5, relief='ridge', width=100, height=200)

        self.thumbnails = ttk.Frame(self.content, relief='sunken', padding=0)
        self.viewer_image = ttk.Label(self.content, text="image", relief='ridge')
        self.viewer_image.config(anchor='center', padding=0)

        self.run_mode = ttk.Combobox(self.options, textvariable=self.run_mode_var)
        self.run_mode['values'] = ('Naive', 'Pyramids')
        self.run_mode.set("Select Mode")

        self.button_dir = ttk.Button(self.options, text="Select folder",
                                     command=lambda: self.load_folder())
        self.path = ttk.Entry(self.options, textvariable=self.image_path, width=50)

        self.button_run = ttk.Button(self.content, text="RUN",
                                     command=lambda: self.run_exposure_fusion())

        self.content.grid(column=0, row=0, sticky='news')
        self.thumbnails.grid(column=1, row=0)
        self.options.grid(column=0, row=0, rowspan=3, sticky='nsew')
        self.viewer_image.grid(column=1, row=1, sticky='news')
        self.run_mode.grid(column=0, row=2, padx=20, sticky='ew')
        self.button_dir.grid(column=0, row=1)
        self.path.grid(column=0, row=0)
        self.button_run.grid(column=0, row=2, sticky='s', pady=(0, 20))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=10)
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=10)

        self.options.columnconfigure(0, weight=1)

        self.thumbnails.columnconfigure(0, weight=1)
        self.thumbnails.columnconfigure(1, weight=1)
        self.thumbnails.columnconfigure(2, weight=1)
        self.thumbnails.rowconfigure(0, weight=1)

        self.root.mainloop()

    def load_folder(self):
        self.image_path.set(filedialog.askdirectory(initialdir='./'))
        self.show_thumbnails()

    def run_exposure_fusion(self):
        self.n_images, images = exposure_fusion.open_images(self.image_path.get())
        fusion = exposure_fusion.ExposureFusion(self.run_mode_var.get().lower(), self.n_images)
        hdr = fusion(images)
        self.save_image(hdr)

    def save_image(self, image: np.ndarray):
        save_file = f"./out/{os.path.split(self.image_path.get())[-1]}-{self.run_mode_var.get().upper()}.jpg"
        # cv.imshow(f'{save_file}',image)
        # cv.waitKey(0)
        cv.imwrite(f"{save_file}", image, [cv.IMWRITE_JPEG_QUALITY, 100])

        self.save_path = save_file
        self.show_image()

    def show_image(self):
        img = Image.open(self.save_path)
        max_width, max_height = self.viewer_image.winfo_width(), self.viewer_image.winfo_height()
        img.thumbnail((max_width, max_height), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        self.viewer_image['image'] = img
        self.viewer_image.image = img

    def show_thumbnails(self):
        file_list = os.listdir(self.image_path.get())
        for i, file in enumerate(file_list):

            img = Image.open(f'{self.image_path.get()}/{file}')
            img.thumbnail((200, 170), Image.ANTIALIAS)
            print(f"{img.width},{img.height}")
            img_tk = ImageTk.PhotoImage(img)
            t = ttk.Label(self.thumbnails, image=img_tk, padding=0, borderwidth=0, relief='ridge')
            t.image = img_tk

            t.grid(column=i, row=0, ipady=0)
            self.thumbnails.config(borderwidth=0)
