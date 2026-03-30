import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    #Logistic Regression
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    #Categorical Naive Bayes
    from sklearn.naive_bayes import CategoricalNB

    #Neural Net
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    #Support Vector Classification
    from sklearn import svm



    return (
        CategoricalNB,
        LogisticRegression,
        MLPClassifier,
        accuracy_score,
        mo,
        os,
        pd,
        svm,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Selector
    """)
    return


@app.cell
def _(os):
    # defining the data folder
    DATA_FOLDER = "../data"

    # list all csv files in folder
    dataset_files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.endswith(".csv")
    ]
    return DATA_FOLDER, dataset_files


@app.cell
def _(dataset_files, mo):
    dataset_selector = mo.ui.dropdown(
        options=dataset_files,
        value=dataset_files[0] if dataset_files else None,
        label="Select Dataset"
    )

    dataset_selector
    return (dataset_selector,)


@app.cell
def _(DATA_FOLDER, dataset_selector, os, pd):
    selected_path = os.path.join(DATA_FOLDER, dataset_selector.value)

    df = pd.read_csv(selected_path)

    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ##Data Preprocessing
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    #Defining Test Dataframe and Processing it.
    df["gender"] = df["gender"].replace({
        "Male": 0,
        "Female": 1,
        "Other": 3
    })

    df['marital_status'] = df["marital_status"].replace({
        "Single": 0,
        "Married": 1,
        "Divorced": 2,
        "Widowed": 3
    })

    df['education_level'] = df["education_level"].replace({
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3,
        "Other":4
    })
    df['employment_status'] = df["employment_status"].replace({
        "Unemployed": 0,
        "Employed": 1,
        "Self-employed": 2,
        "Retired": 3,
        "Student":4
    })
    df['loan_purpose'] = df["loan_purpose"].replace({
        "Debt consolidation": 0,
        "Home": 1,
        "Education": 2,
        "Vacation": 3,
        "Car": 4,
        "Medical": 5,
        "Business": 6,   
        "Other": 7,
    })
    df["grade_subgrade"] = df["grade_subgrade"].replace({
        "F1": 0,
        "F2": 1,
        "F3": 2,
        "F4": 3,
        "F5": 4,
        "E1": 5,
        "E2": 6,
        "E3": 7,
        "E4": 8,
        "E5": 9,
        "D1": 10,
        "D2": 11,
        "D3": 12,
        "D4": 13,
        "D5": 14,
        "C1": 15,
        "C2": 16,
        "C3": 17,
        "C4": 18,
        "C5": 19,
        "B1": 20,
        "B2": 21,
        "B3": 22,
        "B4": 23,
        "B5": 24,
        "A1": 25,
        "A2": 26,
        "A3": 27,
        "A4": 28,
        "A5": 29
    })

    df
    return


@app.cell
def _(df, train_test_split):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(columns=["loan_paid_back"])
    y_train = train_df["loan_paid_back"]

    X_test = test_df.drop(columns=["loan_paid_back"])
    y_test = test_df["loan_paid_back"]
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## Logistic Regression

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """)
    return


@app.cell
def _(LogisticRegression, X_test, X_train, accuracy_score, y_test, y_train):
    log_model = LogisticRegression()

    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_test)

    log_acc = accuracy_score(y_test, log_preds)
    print("Logistic Regression Accuracy:", log_acc)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##Categorical Naive Bayes

    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB
    """)
    return


@app.cell
def _(CategoricalNB, X_test, X_train, accuracy_score, y_test, y_train):
    nb_model = CategoricalNB()

    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)

    nb_acc = accuracy_score(y_test, nb_preds)
    print("Naive Bayes Accuracy:", nb_acc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##kNN; K Nearest Neighbors

    https://scikit-learn.org/stable/modules/neighbors.html
    """)
    return


@app.cell
def _(KNeighborsClassifier, X_test, X_train, accuracy_score, y_test, y_train):
    knn_model = KNeighborsClassifier()

    knn_model.fit(X_train, y_train)
    knn_preds = knn_model.predict(X_test)

    knn_acc = accuracy_score(y_test, knn_preds)
    print("kNN Accuracy:", knn_acc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural Networks

    https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    """)
    return


@app.cell
def _(MLPClassifier, X_test, X_train, accuracy_score, y_test, y_train):
    mlp_model = MLPClassifier()

    mlp_model.fit(X_train, y_train)
    mlp_preds = mlp_model.predict(X_test)

    mlp_acc = accuracy_score(y_test, mlp_preds)
    print("Neural Net Accuracy:", mlp_acc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Support Vector Classification

    https://scikit-learn.org/stable/modules/svm.html
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    """)
    return


@app.cell
def _(X_test, X_train, accuracy_score, svm, y_test, y_train):
    svm_model = svm.SVC()

    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)

    svm_acc = accuracy_score(y_test, svm_preds)
    print("SVM Accuracy:", svm_acc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest Classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    X_test,
    X_train,
    accuracy_score,
    y_test,
    y_train,
):
    rf_model = RandomForestClassifier()

    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    print("Random Forest Accuracy:", rf_acc)
    return


if __name__ == "__main__":
    app.run()
