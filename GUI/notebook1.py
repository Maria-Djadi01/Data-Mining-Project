import tkinter as tk
from tkinter import ttk
import pandas as pd
from PIL import Image, ImageTk

freq_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\Discretized_Distribution_Frequency.png"
width_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\Discretized_Distribution_Width.png"

window_width = 1000
window_height = 650


def show_image(image_path, width, height):
    image = Image.open(image_path)

    # Get the original image size
    original_width, original_height = image.size

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate the new size to fit the canvas while maintaining aspect ratio
    if width / height > aspect_ratio:
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Convert the resized image to Tkinter format
    tk_image = ImageTk.PhotoImage(resized_image)

    # Update the canvas with the resized image
    canvas1.create_image(0, 0, anchor="nw", image=tk_image)
    canvas1.image = tk_image  # Keep a reference to prevent garbage collection


# ------------------------------------------------------------
# Fertility
# ------------------------------------------------------------
notebook = ttk.Notebook(root, width=window_width, height=window_height)
notebook.pack(pady=10, padx=10)

tab3 = ttk.Frame(notebook)

sidebar_frame = ttk.Frame(
    tab3,
    width=window_width / 5,
    height=window_height,
    relief="raised",
    borderwidth=2,
)
sidebar_frame.pack(side="left", fill="y")

# Visualization area
visualization_frame = ttk.Frame(
    tab3,
    width=window_width - window_width / 5,
    height=window_height,
    relief="raised",
    borderwidth=2,
)
visualization_frame.pack(side="left", fill="both")

canvas1 = tk.Canvas(
    visualization_frame, width=window_width - window_width / 5, height=window_height
)
canvas1.pack()

desc_freq_button_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Equal-Frequency Discretization",
    command=lambda: show_image(freq_img, canvas1.winfo_width(), canvas1.winfo_height()),
)
desc_freq_button_button.pack(pady=10)

desc_width_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Equal-Width Discretization",
    command=lambda: show_image(
        width_img, canvas1.winfo_width(), canvas1.winfo_height()
    ),
)
desc_width_button.pack(pady=10)

apriori_entry = tk.Entry(sidebar_frame, width=30)
apriori_entry.pack(pady=10)
apriori_entry.insert(0, "Enter support value")
apriori_entry.config(fg="grey")
apriori_entry.bind("<FocusIn>", lambda args: apriori_entry.delete("0", "end"))
apriori_entry.bind(
    "<FocusOut>", lambda args: apriori_entry.insert(0, "Enter support value")
)

apriori_button = ttk.Button(sidebar_frame, width=30, padding=5, text="apriori Data")
apriori_button.pack(pady=10)
