import tkinter as tk
from tkinter import ttk
from tkinter import *
import pandas as pd
from PIL import Image, ImageTk
import sys
import os


project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_directory)

# sys.path.insert(0, "../../../Data-Mining-Project")
from models.apriori import apriori

# Agriculture
agri_eda_images = {
    "Data": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\df_1.png",
    "Columns": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\info.png",
    "Quantiles": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\descCol_1.png",
    "Histogram": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\histo.png",
    "Box Plot": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\box.png",
    "Scatter Plot": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\scatter.png",
    "Fertility": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\fertilityHist.png",
}

agri_process_images = {
    "Outliers": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\outliers.png",
    "Box Plot": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\no_outliers_box.png",
    "Hitmap": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\features_hitmap.png",
    "Min-Max Normalization": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\minMax_dist.png",
    "Z-score Normalization": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\Dataset_1\zscore_dist.png",
}

# Covid-19
covid_eda_images = {
    "Data": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\df_2.png",
    "Columns": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\info_2.png",
    "Quantiles": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\descCol_2.png",
    "Histogram": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\data2_Hist.png",
    "Bar Plot": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\data2_Bar.png",
}

dist_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\01_distribution_of_the_total_number_of_confirmed_cases_and_positive_tests_by_zones_bar_plot.png"
evol_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\02_evolution_over_time_for_94085.png"
dist2_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\03_distribution_of_positive_covid_cases_by_zone_and_year.png"
scatter_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\04_scatter_plot_of_population_and_test_count.png"
cases_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\05_total_number_of_cases_per_zone.png"
comp_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\02_TSA_2\06_comparison_of_confirmed_cases_tests_conducted_and_positive_tests_for_time_period.png"

# Fertility
fert_eda_images = {
    "Data": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\df_3.png",
    "Columns": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\info_3.png",
    "Quantiles": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\descCol_3.png",
    "Numerical": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\Data Distribution of Numeric Columns.png",
    "Categorical": r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\Data_Distribution_of_Categorical_Columns.png",
}

freq_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\Discretized_Distribution_Frequency.png"
width_img = r"C:\Users\HI\My-Github\Data-Mining-Project\reports\figures\03_EDA\Discretized_Distribution_Width.png"

window_width = 1100
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
    canvas.create_image(0, 0, anchor="nw", image=tk_image)
    canvas.image = tk_image  # Keep a reference to prevent garbage collection


def update_image(selected_choice, choice_images, canvas):
    selected_option = selected_choice.get()

    # Load the selected image
    image_path = choice_images.get(selected_option)
    image = Image.open(image_path)

    # Resize the image to fit the canvas
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    image = image.resize((canvas_width, canvas_height), Image.LANCZOS)

    # Convert the image to Tkinter format
    tk_image = ImageTk.PhotoImage(image)

    # Update the canvas with the resized image
    canvas.create_image(0, 0, anchor="nw", image=tk_image)
    canvas.image = tk_image


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

# Visualization area
visualization_frame = ttk.Frame(
    tab1,
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

# EDA

eda_label_1 = ttk.Label(sidebar_frame, text="Exploratory Data Analysis", font=("", 15))
eda_label_1.pack(pady=10)

agri_eda_choices = [
    "Data",
    "Columns",
    "Quantiles",
    "Histogram",
    "Box Plot",
    "Scatter Plot",
    "Fertility",
]
agri_eda_choice = StringVar()
agri_eda_choice.set(agri_eda_choices[0])
choice_combobox = ttk.Combobox(
    sidebar_frame, textvariable=agri_eda_choice, values=agri_eda_choices
)
choice_combobox.pack()

choice_combobox.bind(
    "<<ComboboxSelected>>",
    lambda event: update_image(agri_eda_choice, agri_eda_images, canvas1),
)

# Preprocessing

process_label_1 = ttk.Label(sidebar_frame, text="Data Pre Processing", font=("", 15))
process_label_1.pack(pady=10)

agri_process_choices = [
    "Outliers",
    "Box Plot",
    "Hitmap",
    "Min-Max Normalization",
    "Z-score Normalization",
]
agri_process_choice = StringVar()
agri_process_choice.set(agri_process_choices[0])
choice_combobox = ttk.Combobox(
    sidebar_frame, textvariable=agri_process_choice, values=agri_process_choices
)
choice_combobox.pack()

choice_combobox.bind(
    "<<ComboboxSelected>>",
    lambda event: update_image(agri_process_choice, agri_process_images, canvas1),
)

k_entry = tk.Entry(sidebar_frame, width=30)
k_entry.pack(pady=10)
k_entry.insert(0, "Enter Number of Neighbors")
k_entry.config(fg="grey")
k_entry.bind("<FocusIn>", lambda event, e=k_entry: on_entry_click(e, "K Value"))
k_entry.bind("<FocusOut>", lambda event, e=k_entry: on_focus_out(e, "K Value"))


import time
import sys
import io
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

sys.path.insert(0, "../../../Data-Mining-Project")
from models.knn import KNN
from src.utils import split_data, compute_metrics, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


df = pd.read_csv(
    r"C:\Users\HI\My-Github\Data-Mining-Project\data\interim\03_static_dataset_features_built.csv",
    index_col=0,
)
X_train, X_test, y_train, y_test = split_data(df)
desired_num_samples = 34
sampling_strategy_dict = {
    class_label: desired_num_samples
    for class_label, desired_num_samples in zip(*np.unique(y_train, return_counts=True))
}

undersampler = RandomUnderSampler(
    sampling_strategy=sampling_strategy_dict, random_state=42
)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)


def execute_knn(KNN, k, canvas):
    knn_3 = KNN(k)
    start_time = time.time()
    knn_3.fit(X_resampled, y_resampled)
    y_pred = knn_3.predict(X_test)
    end_time = time.time()
    RF_exec_time = end_time - start_time
    metrics_result = compute_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Format the information
    info_text += f"Metrics:\n"
    info_text += f"Accuracy: {metrics_result['accuracy']:.2f}\n"
    info_text += f"Precision: {metrics_result['precision']:.2f}\n"
    info_text += f"Recall: {metrics_result['recall']:.2f}\n"
    info_text += f"F1 Score: {metrics_result['f1_score']:.2f}\n"
    info_text += f"Specificity: {metrics_result['specificity']:.2f}\n"
    info_text += f"Execution Time: {RF_exec_time:.2f} seconds"

    # Clear existing widgets in the canvas
    for widget in canvas.winfo_children():
        widget.destroy()

    # Create or update a text widget in the canvas
    text_widget = tk.Text(canvas, wrap="word", width=40, height=10)
    text_widget.insert(tk.END, info_text)
    text_widget.pack()
    plot_confusion_matrix(cm)


knn_button = ttk.Button(
    sidebar_frame,
    width=30,
    padding=5,
    text="Knn",
    command=lambda: execute_knn(KNN, int(k_entry.get()), canvas1),
)
knn_button.pack(pady=10)

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

# EDA

eda_label_2 = ttk.Label(sidebar_frame, text="Exploratory Data Analysis", font=("", 15))
eda_label_2.pack(pady=10)

covid_eda_choices = ["Data", "Columns", "Quantiles", "Histogram", "Bar Plot"]
covid_eda_choice = StringVar()
covid_eda_choice.set(covid_eda_choices[0])
choice_combobox = ttk.Combobox(
    sidebar_frame, textvariable=covid_eda_choice, values=covid_eda_choices
)
choice_combobox.pack()

choice_combobox.bind(
    "<<ComboboxSelected>>",
    lambda event: update_image(covid_eda_choice, covid_eda_images, canvas2),
)

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

# EDA

eda_label_3 = ttk.Label(sidebar_frame, text="Exploratory Data Analysis", font=("", 15))
eda_label_3.pack(pady=10)

fert_eda_choices = ["Data", "Columns", "Quantiles", "Numerical", "Categorical"]
fert_eda_choice = StringVar()
fert_eda_choice.set(fert_eda_choices[0])
choice_combobox = ttk.Combobox(
    sidebar_frame, textvariable=fert_eda_choice, values=fert_eda_choices
)
choice_combobox.pack()

choice_combobox.bind(
    "<<ComboboxSelected>>",
    lambda event: update_image(fert_eda_choice, fert_eda_images, canvas3),
)

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
