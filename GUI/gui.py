import tkinter as tk
from tkinter import ttk
import pandas as pd
from PIL import Image, ImageTk
import sys
import os


project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_directory)

# sys.path.insert(0, "../../../Data-Mining-Project")
from models.apriori import apriori

freq_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\Discretized_Distribution_Frequency.png"
width_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\Discretized_Distribution_Width.png"
dist_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\01_distribution_of_the_total_number_of_confirmed_cases_and_positive_tests_by_zones_bar_plot.png"
evol_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\02_evolution_over_time_for_94085.png"
dist2_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\03_distribution_of_positive_covid_cases_by_zone_and_year.png"
scatter_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\04_scatter_plot_of_population_and_test_count.png"
cases_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\05_total_number_of_cases_per_zone.png"
comp_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\06_comparison_of_confirmed_cases_tests_conducted_and_positive_tests_for_time_period.png"
window_width = 1000
window_height = 650

df = pd.read_csv(
    r"C:\Users\HI\My-Github\Data-Mining-Project\data\processed\static_dataset3_discretized.csv",
    index_col=0,
)

apriori_df = df.drop(
    columns=[
        "Temperature",
        "Humidity",
        "Rainfall",
        "Temperature_width_disc",
        "Humidity_width_disc",
        "Rainfall_width_disc",
    ]
)


def show_image(image_path, canvas):
    image = Image.open(image_path)
    width, height = canvas.winfo_width(), canvas.winfo_height()
    print(width, height)

    # Get the original image size
    original_width, original_height = image.size

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height
    print(aspect_ratio)
    print(width / height)

    # Calculate the new size to fit the canvas while maintaining aspect ratio
    if width / height > aspect_ratio:
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / aspect_ratio)
    print(new_width, new_height)
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Convert the resized image to Tkinter format
    tk_image = ImageTk.PhotoImage(resized_image)

    # Update the canvas with the resized image
    canvas.create_image(0, 0, anchor="nw", image=tk_image)
    canvas.image = tk_image  # Keep a reference to prevent garbage collection


def on_entry_click(entry, placeholder):
    if entry.get() == placeholder:
        entry.delete(0, "end")  # Delete the default placeholder text
        entry.insert(0, "")  # Set the text color to black


def on_focus_out(entry, placeholder):
    if entry.get() == "":
        entry.insert(0, placeholder)  # If no text was entered, set placeholder back
        entry.config(fg="grey")


# Create the main window
root = tk.Tk()
root.title("Data Mining Project")


# Create and configure the notebook (tabbed interface)
notebook = ttk.Notebook(root, width=window_width, height=window_height)
notebook.pack(pady=10, padx=10)

# ------------------------------------------------------------
# AGRICULTURE
# ------------------------------------------------------------
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Agriculture")

sidebar_frame = ttk.Frame(
    tab1,
    width=window_width / 5,
    height=window_height,
    relief="raised",
    borderwidth=2,
)
sidebar_frame.pack(side="left", fill="y")

EDA_button = ttk.Button(sidebar_frame, width=30, padding=5, text="EDA Data")
EDA_button.pack(pady=10)

analyze_button = ttk.Button(sidebar_frame, width=30, padding=5, text="Analyze Data")
analyze_button.pack(pady=10)

analyze_button = ttk.Button(sidebar_frame, width=30, padding=5, text="Analyze Data")
analyze_button.pack(pady=10)

# Visualization area
visualization_frame = ttk.Frame(
    tab1,
    width=window_width - window_width / 5,
    height=window_height,
    relief="raised",
    borderwidth=2,
)
visualization_frame.pack(side="left", fill="both")

# ------------------------------------------------------------
# COVID-19
# ------------------------------------------------------------
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Covid-19")

sidebar_frame = ttk.Frame(
    tab2,
    width=window_width / 5,
    height=window_height,
    relief="raised",
    borderwidth=2,
)
sidebar_frame.pack(side="left", fill="y")

# Visualization area
visualization_frame = ttk.Frame(
    tab2,
    width=window_width - window_width / 5,
    height=window_height,
    relief="raised",
    borderwidth=2,
)
visualization_frame.pack(side="left", fill="both")

canvas2 = tk.Canvas(
    visualization_frame, width=window_width - window_width / 5, height=window_height
)
canvas2.pack()

totalNum_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Distribution by zones",
    command=lambda: show_image(dist_img, canvas2),
)
totalNum_button.pack(pady=10)

evol_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Evolution by zones",
    command=lambda: show_image(evol_img, canvas2),
)
evol_button.pack(pady=10)

posCase_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Positive Cases Distribution",
    command=lambda: show_image(dist2_img, canvas2),
)
posCase_button.pack(pady=10)

relation_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Population vs Conducted Tests",
    command=lambda: show_image(scatter_img, canvas2),
)
relation_button.pack(pady=10)

cases_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Zones Impacted by Covid",
    command=lambda: show_image(cases_img, canvas2),
)
cases_button.pack(pady=10)

comp_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Confirmed Cases vs Positive Tests",
    command=lambda: show_image(comp_img, canvas2),
)
comp_button.pack(pady=10)

# ------------------------------------------------------------
# Fertility
# ------------------------------------------------------------
tab3 = ttk.Frame(notebook)
notebook.add(tab3, text="Fertility")

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

canvas3 = tk.Canvas(
    visualization_frame, width=window_width - window_width / 5, height=window_height
)
canvas3.pack()

desc_freq_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Equal-Frequency Discretization",
    command=lambda: show_image(freq_img, canvas3),
)
desc_freq_button.pack(pady=10)

desc_width_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Equal-Width Discretization",
    command=lambda: show_image(width_img, canvas3),
)
desc_width_button.pack(pady=10)

minSup_entry = tk.Entry(sidebar_frame, width=30)
minSup_entry.pack(pady=10)
minSup_entry.insert(0, "Enter support value")
minSup_entry.config(fg="grey")
minSup_entry.bind(
    "<FocusIn>", lambda event, e=minSup_entry: on_entry_click(e, "Support Value")
)
minSup_entry.bind(
    "<FocusOut>", lambda event, e=minSup_entry: on_focus_out(e, "Support Value")
)

minConf_entry = tk.Entry(sidebar_frame, width=30)
minConf_entry.pack(pady=10)
minConf_entry.insert(0, "Enter support value")
minConf_entry.config(fg="grey")
minConf_entry.bind(
    "<FocusIn>", lambda event, e=minConf_entry: on_entry_click(e, "Confidence Value")
)
minConf_entry.bind(
    "<FocusOut>", lambda event, e=minConf_entry: on_focus_out(e, "Confidence Value")
)

apriori_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Apriori",
    command=lambda: apriori(
        df, apriori_df, int(minSup_entry.get()), int(minConf_entry.get())
    ),
)
apriori_button.pack(pady=10)

# ------------------------------------------------------------

root.geometry(f"{window_width}x{window_height}")
root.mainloop()
