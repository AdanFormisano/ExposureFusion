import os
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import exposure_fusion
from PIL import ImageTk, Image
import cv2 as cv
import time


class GUI:
    def __init__(self):
        self.files_wrong_ext = []
        self.time_elapsed = None

        self.root = Tk()
        self.root.geometry("800x500")
        self.root.title("Exposure Fusion")
        self.root.resizable(False, False)

        self.run_mode_var = StringVar()
        self.image_path = StringVar()
        self.n_images = None
        self.save_path = None
        self.viewer_image = None
        self.thumbnails = None

        self.content = ttk.Frame(self.root)

        self.thumbnails = ttk.Frame(self.content)
        self.viewer_image = ttk.Label(self.content)
        self.thumbnails.grid(column=1, row=0, columnspan=3, sticky='news')
        self.viewer_image.grid(column=1, row=1, columnspan=3, rowspan=2)

        self.options = ttk.Frame(self.content)

        self.run_mode = ttk.Combobox(self.options, textvariable=self.run_mode_var)
        self.run_mode['values'] = ('Naive', 'Pyramids')
        self.run_mode.set("Select Mode")

        self.button_dir = ttk.Button(self.options, text="Select folder",
                                     command=lambda: self.load_folder())
        self.path = ttk.Entry(self.options, textvariable=self.image_path)

        self.button_run = ttk.Button(self.content, text="RUN",
                                     command=lambda: self.run_exposure_fusion())

        # Grid geometry manager
        self.content.grid(column=0, row=0, sticky='news')

        self.options.grid(column=0, row=0, rowspan=2, sticky='nsew')
        self.button_dir.grid(column=1, row=0, pady=10, padx=(5, 10))
        self.path.grid(column=0, row=0, pady=10, padx=(10, 5), sticky='we')
        self.run_mode.grid(column=0, row=1, pady=10, padx=10, sticky='w')

        self.button_run.grid(column=0, row=2, pady=10)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=5)
        self.content.columnconfigure(2, weight=5)
        self.content.columnconfigure(3, weight=5)
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)

        self.thumbnails.columnconfigure(0, weight=1)
        self.thumbnails.columnconfigure(1, weight=1)
        self.thumbnails.columnconfigure(2, weight=1)
        self.thumbnails.rowconfigure(0, weight=1)

        self.options.columnconfigure(0, weight=5)
        # self.options.rowconfigure(0, weight=1)
        # self.options.rowconfigure(1, weight=1)
        # self.options.rowconfigure(2, weight=1)

        self.root.mainloop()

    def check_ext(self):
        for file in os.listdir(self.image_path.get()):
            if not file.endswith('.jpg'):
                self.files_wrong_ext.append(file)

    def load_folder(self):
        self.image_path.set(filedialog.askdirectory(initialdir='./'))
        self.check_ext()

        if not len(os.listdir(self.image_path.get())) >= 3:
            messagebox.showinfo(message='ERROR: Not enough images!\nPlease select a folder that has at least 3 images.',
                                title="ERROR", icon="error")

        elif self.files_wrong_ext:
            print(f"{self.files_wrong_ext}")
            messagebox.showinfo(message=f"ERROR: {*self.files_wrong_ext,} are not JPG!"
                                        f" Please select a folder with only JPG images.",
                                title="ERROR", icon="error")
            self.files_wrong_ext = []
        else:
            self.thumbnails.forget()
            self.thumbnails = ttk.Frame(self.content)
            self.thumbnails.grid(column=1, row=0, columnspan=3, sticky='news')
            self.thumbnails.grid_propagate(False)
            self.show_thumbnails()

    def run_exposure_fusion(self):
        self.n_images, images = exposure_fusion.open_images(self.image_path.get())

        start_time = time.perf_counter()

        fusion = exposure_fusion.ExposureFusion(self.run_mode_var.get().lower(), self.n_images)
        hdr = fusion(images)

        self.time_elapsed = time.perf_counter() - start_time
        self.viewer_image.destroy()
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
        max_width, max_height = 450, 540
        img.thumbnail((max_width, max_height), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        self.viewer_image = ttk.Label(self.content, relief='sunken')
        self.viewer_image.config(anchor='center', padding=0, borderwidth=0)
        self.viewer_image.grid(column=1, row=1, columnspan=3, rowspan=2)

        self.viewer_image['image'] = img
        self.viewer_image.image = img

        messagebox.showinfo(message=f'Done in {round(self.time_elapsed, 3)} seconds!', title="Exposure Fusion finished",
                            icon="info")

    def show_thumbnails(self):
        file_list = os.listdir(self.image_path.get())
        for i, file in enumerate(file_list):
            img = Image.open(f'{self.image_path.get()}/{file}')
            img.thumbnail((100, 180), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img)
            t = ttk.Label(self.thumbnails, relief='ridge')
            t['image'] = img_tk
            t.image = img_tk
            t.grid(column=i, row=0)

            self.thumbnails.columnconfigure(i, weight=1)
