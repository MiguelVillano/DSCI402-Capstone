import tkinter as tk
from tkinter import ttk
import pandas as pd
from PIL import Image, ImageTk

# -----------------------------
# datasets
# -----------------------------
datasets = {
    "Trained": "data/train.csv",
    "Cleaned": "data/train_clean.csv",
    "Log Transform": "data/train_clean_logged.csv",
    "SQRT Transform": "data/train_clean_sqrt.csv",
    "BoxCox Transform": "data/train_clean_boxcox.csv",
    "YeoJohnson Transform": "data/train_clean_yeojohnson.csv",
}

# -----------------------------
# histogram images
# -----------------------------
images = {
    "Trained": "images/trainData.png",
    "Cleaned": "images/trainDataCleaned.png",
    "Log Transform": "images/images/log_annual_income_histogram.png",
    "SQRT Transform": "images/sqrt_annual_income_histogram.png",
    "BoxCox Transform": "images/boxcox_annual_income_histogram.png",
    "YeoJohnson Transform": "images/yeojohnson_annual_income_histogram.png",
}

df = None


# -----------------------------
# update table display
# -----------------------------
def update_table(data):
    info_text.set(
        f"Rows: {data.shape[0]}\n"
        f"Columns: {data.shape[1]}\n"
        f"Column Names: {', '.join(data.columns)}"
    )

    for row in table.get_children():
        table.delete(row)

    table["columns"] = list(data.columns)
    table["show"] = "headings"

    for col in data.columns:
        table.heading(col, text=col)
        table.column(col, width=120)

    for _, row in data.head().iterrows():
        table.insert("", "end", values=list(row))


# -----------------------------
# load dataset
# -----------------------------
def load_dataset():
    global df
    dataset_name = dataset_var.get()
    if dataset_name == "":
        return
    file_path = datasets[dataset_name]
    df = pd.read_csv(file_path)
    sort_column_dropdown["values"] = list(df.columns)
    update_table(df)


# -----------------------------
# sort dataset
# -----------------------------
def sort_data(ascending=True):
    if df is None:
        return
    column = sort_column_var.get()
    if column == "":
        return
    sorted_df = df.sort_values(by=column, ascending=ascending)
    update_table(sorted_df)


# -----------------------------
# show histogram image
# -----------------------------
def show_distribution():
    dataset_name = dataset_var.get()
    if dataset_name == "":
        return
    img_path = images[dataset_name]
    img = Image.open(img_path)
    img = img.resize((750, 450))
    global img_tk
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)


# -----------------------------
# window
# -----------------------------
root = tk.Tk()
root.title("Loan Dataset Explorer")
root.state('zoomed')

# -----------------------------
# dataset selector
# -----------------------------
selector_frame = ttk.Frame(root)
selector_frame.pack(pady=10)

ttk.Label(selector_frame, text="Select Dataset:").grid(row=0, column=0, padx=5)

dataset_var = tk.StringVar()

dataset_dropdown = ttk.Combobox(selector_frame, textvariable=dataset_var)
dataset_dropdown["values"] = list(datasets.keys())
dataset_dropdown.grid(row=0, column=1, padx=5)

enter_button = ttk.Button(selector_frame, text="Enter", command=load_dataset)
enter_button.grid(row=0, column=2, padx=5)

# -----------------------------
# sorting controls
# -----------------------------
sort_frame = ttk.Frame(root)
sort_frame.pack(pady=10)

ttk.Label(sort_frame, text="Order By:").grid(row=0, column=0, padx=5)

sort_column_var = tk.StringVar()

sort_column_dropdown = ttk.Combobox(sort_frame, textvariable=sort_column_var)
sort_column_dropdown.grid(row=0, column=1, padx=5)

smallest_button = ttk.Button(
    sort_frame,
    text="Smallest",
    command=lambda: sort_data(True)
)
smallest_button.grid(row=0, column=2, padx=5)

largest_button = ttk.Button(
    sort_frame,
    text="Largest",
    command=lambda: sort_data(False)
)
largest_button.grid(row=0, column=3, padx=5)

# -----------------------------
# show distribution button
# -----------------------------
plot_button = ttk.Button(root, text="Show Distribution", command=show_distribution)
plot_button.pack(pady=10)

# -----------------------------
# dataset info
# -----------------------------
info_text = tk.StringVar()

info_label = ttk.Label(root, textvariable=info_text, justify="left")
info_label.pack(pady=10)

# -----------------------------
# table
# -----------------------------
table_frame = ttk.Frame(root)
table_frame.pack(fill="both", expand=True)

scrollbar = ttk.Scrollbar(table_frame)
scrollbar.pack(side="right", fill="y")

table = ttk.Treeview(table_frame, yscrollcommand=scrollbar.set)
table.pack(fill="both", expand=True)

scrollbar.config(command=table.yview)

# -----------------------------
# histogram display
# -----------------------------
image_label = ttk.Label(root)
image_label.pack(pady=15)

# keyboard enter loads dataset
root.bind("<Return>", lambda event: load_dataset())

root.mainloop()