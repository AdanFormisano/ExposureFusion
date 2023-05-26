import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

def load_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_paths:
        for file_path in file_paths:
            image = Image.open(file_path)
            image = image.resize((300, 200))  # Resize the image as desired
            photo = ImageTk.PhotoImage(image)
            image_labels.append(tk.Label(window, image=photo))
            image_labels[-1].image = photo  # Store a reference to the image to prevent it from being garbage collected
            image_labels[-1].pack(pady=10)

# Create a new instance of Tkinter window
window = tk.Tk()

# Set the window title
window.title("Image Loader")

# Set the window dimensions
window.geometry("400x300")

# Create a button to load images
load_button = tk.Button(window, text="Load Images", command=load_images)
load_button.pack(pady=10)

# Create a list to store the image labels
image_labels = []

# Run the Tkinter event loop
window.mainloop()
