import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    TO DO NEXT TIME:

    Git Repo + Git Commits
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
    path = 'test.csv'
    df = pd.read_csv(path)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    Missing Data Investigation
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
    Outlier Detection
    """)
    return


@app.cell
def _():
    #Scan for Outliers here
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
    ### Feature Visualization
    A quick analysis on the data through plots and correlation investigation. This could give an insight as to which features are important in magnitude or trends.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Processing
    In order to use the data for the models we plan to use, many of the categorical variables must be turned numerical, binary or ordinal.
    """)
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
