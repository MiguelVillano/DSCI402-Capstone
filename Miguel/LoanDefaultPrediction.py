import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    TO DO NEXT TIME:

    Git Repo + Git Commits DONE
    Process Data into numerical values
    Do Visualization and Correlation Investigation
    Maybe start on Logistic Regression???
    """)
    return


@app.cell
def load_libraries():
    import subprocess, sys

    # packages we need for this project
    packages = [
        "marimo",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
    ]
    
    # import each package, but if there is an error than it will pip install
    for pkg in packages:
        try:
            __import__(pkg if pkg != "scikit-learn" else "sklearn")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    # import
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import preprocessing

    # return the abbreviations
    return mo, pd, np, plt, sns, preprocessing


@app.cell
def _(mo):
    mo.md(r"""
    # LOAN DEFAULT PREDICTION

    https://www.kaggle.com/competitions/playground-series-s5e11/data?select=test.csv
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Investigation:
    - Be able to predict Loan Defaults.
    - What factors are the most important in predicting Loan Defaults?
    - Which models are the most successful in Predictions
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Business Problem Understanding
    [README STYLE OVERVIEW]

    More Information about columns can be found here:
    https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025/data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Understanding
    """)
    return


@app.cell
def _(pd):
    path = 'train.csv'
    df = pd.read_csv(path)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Missing Data Investigation
    """)
    return


@app.cell
def _():
    #df.null().count()

    #This returns Attribute errors as the dataframe contains no null data. 
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature Visualization
    A quick analysis on the data through plots and correlation investigation. This could give an insight as to which features are important in magnitude or trends.
    """)
    return


@app.cell
def _(df, plt):
    #Binary Comparisons 
    counts_gender = df["gender"].value_counts()
    plt.figure()
    plt.bar(counts_gender.index, counts_gender.values)
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Distribution of Gender")
    plt.show()
    return


@app.cell
def _(df, plt):
    counts_marriage = df["marital_status"].value_counts()
    plt.figure()
    plt.bar(counts_marriage.index, counts_marriage.values)
    plt.xlabel("Marital Status")
    plt.ylabel("Count")
    plt.title("Distribution of Marital Status")
    plt.show()
    return


@app.cell
def _(df, plt):
    #Binary Comparisons 
    counts_defaults = df["loan_paid_back"].value_counts()
    plt.figure()
    plt.bar(counts_defaults.index, counts_defaults.values)
    plt.xlabel("Loan Defaults")
    plt.ylabel("Count")
    plt.title("Default Ratio")
    plt.show()
    return


@app.cell
def _(df, plt):
    plt.figure()
    plt.hist(df["annual_income"], bins=100)
    plt.xlim(0, df["annual_income"].max())
    plt.xlabel("Income")
    plt.ylabel("Frequency")

    plt.title("Histogram of incomes")
    return


@app.cell
def _(df):
    df["annual_income"].max()

    #This maximum is a heavy outlier. There is some outliers we will eliminate.
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Correlation Investigation

    Based on domain knowledge, there are some easily
    """)
    return


@app.cell
def _(df):
    r = df["x"].corr(df["y"])
    print("Pearson r:", r)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Outlier Detection

    I will eliminate any outliers as determined by Tukey's Fences
    """)
    return


@app.cell
def _(df):
    value = df.annual_income

    # Calculate quartiles
    Q1 = value.quantile(0.25)
    Q3 = value.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate Tukey's fences
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Lower Bound (Tukey's Fence): {lower_bound}")
    print(f"Upper Bound (Tukey's Fence): {upper_bound}")

    # Identify outliers using Tukey's fences
    tukey_outliers = df[(value < lower_bound) | (value > upper_bound)]

    print("\nOutliers (Tukey's Fences):")
    print(tukey_outliers)
    return


@app.cell
def _(mo):
    mo.md(r"""
    There are many rows deemed as outliers, therefore it is unwise to drop any. In addition, domain understanding importance to some of these values as banks deal with both the richest and the poorest, so any of these outliers could provide useful information to the model. Through feature visualization, there may be outstanding outliers that can be seen, and later model evaluations could provide insight on which feature needs to be normalized.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Due to the Synthetic Nature of the Data, there is no need for Data Cleanup.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Processing
    In order to use the data for the models we plan to use, many of the categorical variables must be turned numerical, binary or ordinal.
    """)
    return


@app.cell
def _(df):
    df["gender"] = df["gender"].replace({
        "male": 0,
        "female": 1,
        "other": 3
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modeling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Result Analysis
    """)
    return


if __name__ == "__main__":
    app.run()
