import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    path = '../data/train_clean_boxcox.csv'
    df = pd.read_csv(path)
    return df, mo, svm, train_test_split


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
    Y_train = train_df["loan_paid_back"]

    X_test = test_df.drop(columns=["loan_paid_back"])
    Y_test = test_df["loan_paid_back"]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Logistic Regression

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##Categorical Naive Bayes

    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##kNN; K Nearest Neighbors

    https://scikit-learn.org/stable/modules/neighbors.html
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural Networks

    https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    """)
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
def _(SVC, X, svm, y):
    clf = svm.SVC()
    clf.fit(X, y)
    SVC()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest Classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
