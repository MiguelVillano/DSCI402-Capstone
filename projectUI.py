import tkinter as tk
from tkinter import ttk, PhotoImage
import pandas as pd
from PIL import Image, ImageTk

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

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
    "Log Transform": "images/log_annual_income_histogram.png",
    "SQRT Transform": "images/sqrt_annual_income_histogram.png",
    "BoxCox Transform": "images/boxcox_annual_income_histogram.png",
    "YeoJohnson Transform": "images/yeojohnson_annual_income_histogram.png",
}

# globals
df = None
original_df = None

# -----------------------------
# update table
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
        table.column(col, width=150, stretch=True)

    for _, row in data.head(50).iterrows():
        table.insert("", "end", values=list(row))

# -----------------------------
# load dataset
# -----------------------------
def load_dataset():
    global df, original_df

    dataset_name = dataset_var.get()
    if dataset_name == "":
        return

    try:
        file_path = datasets[dataset_name]
        df = pd.read_csv(file_path)
        original_df = df.copy()
    except Exception as e:
        info_text.set(f"Error loading dataset: {e}")
        return

    sort_column_dropdown["values"] = list(df.columns)

    # ONLY allow loan_paid_back
    if "loan_paid_back" in df.columns:
        target_dropdown["values"] = ["loan_paid_back"]
        target_var.set("loan_paid_back")
    else:
        target_dropdown["values"] = []
        target_var.set("")

    update_table(df)

# -----------------------------
# sort data
# -----------------------------
def sort_data(ascending=True):
    global df

    if df is None:
        return

    column = sort_column_var.get()
    if column == "":
        return

    df = df.sort_values(by=column, ascending=ascending)
    update_table(df)

# -----------------------------
# reset data
# -----------------------------
def reset_data():
    global df
    if original_df is not None:
        df = original_df.copy()
        update_table(df)

# -----------------------------
# show histogram
# -----------------------------
def show_distribution():
    dataset_name = dataset_var.get()
    if dataset_name == "":
        return

    try:
        img_path = images[dataset_name]
        img = Image.open(img_path)
        img = img.resize((550, 350))

        global img_tk
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
    except Exception as e:
        info_text.set(f"Error loading image: {e}")

# -----------------------------
# popup
# -----------------------------
def show_results_popup(results):
    popup = tk.Toplevel(root)
    popup.title("Model Results")
    popup.geometry("500x400")

    ttk.Label(popup, text="Model Comparison", font=("Arial", 14, "bold")).pack(pady=10)

    text_frame = ttk.Frame(popup)
    text_frame.pack(fill="both", expand=True, padx=10, pady=10)

    scrollbar = ttk.Scrollbar(text_frame)
    scrollbar.pack(side="right", fill="y")

    text_box = tk.Text(text_frame, yscrollcommand=scrollbar.set, wrap="word")
    text_box.pack(fill="both", expand=True)

    scrollbar.config(command=text_box.yview)

    text_box.insert("1.0", results)
    text_box.config(state="disabled")

    try:
        img_path = "images/loan_paid_back_distribution.png"
        img = Image.open(img_path)
        img = img.resize((550, 350))

        global img_tk
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
    except Exception as e:
        print(f"Error loading image: {e}")
        info_text.set(f"Error loading image: {e}")
    
    ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)

# -----------------------------
# remapping numbers
# -----------------------------
def preprocess_data(df, target):
    df = df.copy()

    # separate features + target
    X = df.drop(columns=[target])
    y = df[target]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # REMAP (one-hot encoding)
    # -----------------------------
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # align test to train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # -----------------------------
    # SCALE
    # -----------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

# -----------------------------
# neural net
# -----------------------------
def neural_net(y_train, y_test, X_train_scaled, X_test_scaled):
    # figure out activiation function ????????
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        early_stopping=True,
        random_state=42
    )

    nn.fit(X_train_scaled, y_train)

    nn_preds = nn.predict(X_test_scaled)
    nn_acc = accuracy_score(y_test, nn_preds)
    
    return nn_acc

# -----------------------------
# logistic regression
# -----------------------------
def log_reg(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, preds)
    
    return lr_acc

# -----------------------------
# decision tree
# -----------------------------
def decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, preds)
    
    return dt_acc

# -----------------------------
# train model
# -----------------------------
def train_model():
    global df
    
    if df is None:
        print('No dataset found')
        model_result.set('No dataset found')
        return

    target = target_var.get()
    if target == "":
        print('No target variable')
        model_result.set('No target variable')
        return

    print(f"df:\n{df}")
    print(f"target var: {target}")

    try:
        results = ""
        
        # -----------------------------
        # PREPROCESS
        # -----------------------------
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_data(df, target)

        # -----------------------------
        # NEURAL NETWORK
        # -----------------------------
        nn_acc = neural_net(y_train, y_test, X_train_scaled, X_test_scaled)       
        results += f"Neural Network Accuracy: {nn_acc:.4f}\n"

        # -----------------------------
        # LOGISTIC REGRESSION
        # -----------------------------
        lr_acc = log_reg(X_train, X_test, y_train, y_test)
        results += f"Logistic Regression Accuracy: {lr_acc:.4f}\n"
        
        # -----------------------------
        # DECISION TREE
        # -----------------------------
        dt_acc = decision_tree(X_train, X_test, y_train, y_test)
        results += f"Decision Tree Accuracy: {dt_acc:.4f}\n"

        # -----------------------------
        # SHOW POPUP
        # -----------------------------
        show_results_popup(results)

    except Exception as e:
        print(f"Error:\n{e}")
        model_result.set(f"Error: {e}")

# -----------------------------
# UI
# -----------------------------
root = tk.Tk()
root.title("Loan Dataset Explorer + ML")
root.state('zoomed')

# dataset selector
selector_frame = ttk.Frame(root)
selector_frame.pack(pady=10)

ttk.Label(selector_frame, text="Select Dataset:").grid(row=0, column=0, padx=5)

dataset_var = tk.StringVar()

dataset_dropdown = ttk.Combobox(selector_frame, textvariable=dataset_var)
dataset_dropdown["values"] = list(datasets.keys())
dataset_dropdown.grid(row=0, column=1, padx=5)

enter_button = ttk.Button(selector_frame, text="Enter", command=load_dataset)
enter_button.grid(row=0, column=2, padx=5)

# sorting
sort_frame = ttk.Frame(root)
sort_frame.pack(pady=10)

ttk.Label(sort_frame, text="Order By:").grid(row=0, column=0, padx=5)

sort_column_var = tk.StringVar()

sort_column_dropdown = ttk.Combobox(sort_frame, textvariable=sort_column_var)
sort_column_dropdown.grid(row=0, column=1, padx=5)

smallest_button = ttk.Button(sort_frame, text="Smallest", command=lambda: sort_data(True))
smallest_button.grid(row=0, column=2, padx=5)

largest_button = ttk.Button(sort_frame, text="Largest", command=lambda: sort_data(False))
largest_button.grid(row=0, column=3, padx=5)

reset_button = ttk.Button(sort_frame, text="Reset", command=reset_data)
reset_button.grid(row=0, column=4, padx=5)

# ML section
ml_frame = ttk.Frame(root)
ml_frame.pack(pady=15)

ttk.Label(ml_frame, text="Target Column:").grid(row=0, column=0, padx=5)

target_var = tk.StringVar()

target_dropdown = ttk.Combobox(ml_frame, textvariable=target_var, state="readonly")
target_dropdown.grid(row=0, column=1, padx=5)

train_button = ttk.Button(ml_frame, text="Train Model", command=train_model)
train_button.grid(row=0, column=2, padx=5)

model_result = tk.StringVar()
ttk.Label(ml_frame, textvariable=model_result).grid(row=0, column=3, padx=10)

# plot button
plot_button = ttk.Button(root, text="Show Distribution", command=show_distribution)
plot_button.pack(pady=10)

# info
info_text = tk.StringVar()
ttk.Label(root, textvariable=info_text, justify="left").pack(pady=10)

# table
table_frame = ttk.Frame(root)
table_frame.pack(fill="both", expand=True)

scrollbar = ttk.Scrollbar(table_frame)
scrollbar.pack(side="right", fill="y")

table = ttk.Treeview(table_frame, yscrollcommand=scrollbar.set)
table.pack(fill="both", expand=True)

scrollbar.config(command=table.yview)

# image
image_label = ttk.Label(root)
image_label.pack(pady=15)

# key bind
root.bind("<Return>", lambda event: load_dataset())

root.mainloop()