import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
<<<<<<< HEAD
def _():
=======
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
>>>>>>> 75647186436a72a89604480adb59c2da401a7b94
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    #Logistic Regression
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    return LogisticRegression, metrics, mo, pd, plt, train_test_split


@app.cell
def _(mo):
    mo.md(r"""
    TO ASK DR. KABERNATHY
    Ask about how to turn employment status, loan reason etc. into ordinal data.
    Ask about how to use testing data with no loan_paid_back column
    TO DO NEXT TIME:

    OPTIMIZE AND TEST AROUND THE LOGISTIC REGRESSION
    ROC AUC TEST FOR LOG REG
    START ON NAIVE BAYES
    """)
    return


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

    This project focuses on building a predictive model to estimate likelihood of Loan Defaults using Customer Demographic, and historical financial data. Key features include credit scores, loan amounts, interest rate, gender, marital status and education. The project closely follows the CRISP-DM methodology for Data Science Projects, which include understanding the problem, understanding the data, processing and cleaning it, analyzing results and determining a conclusion. The project involves data preprocessing steps such as handling missing values and outliers, as well as transforming categorical variables into numerical formats suitable for model training. The models used include statistical classifiers such as Logistic Regression and Naive Bayes and machine learning algorithms including K-Nearest Neighbors, Neural Networks, and Support Vector Machines. Model performance is evaluated by testing against a testing set, split from the training set, and checking its accuracy. The final model will provide insights on which features, or combination thereof, can anticipate a Loan Default.

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
    path = '../train.csv'
    df = pd.read_csv(path)
    return df, path


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.max()
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

    Based on domain knowledge, there are some features that we can guess to be important to defaults, such as annual income, education and credit score.
    """)
    return


@app.cell
def _(df):
    r_income = df["annual_income"].corr(df["loan_paid_back"])
    print("Annual Income Correlation:", r_income)

    print("The result here indicates that a higher annual income leads to a higher chance of paying back the loan.")
    return


@app.cell
def _(df):
    r_creditscore = df["credit_score"].corr(df["loan_paid_back"])
    print("Credit Score Correlation:", r_creditscore)

    print("It seems there is a little correlation between a high credit score and not defaulting.")
    return


@app.cell
def _(df):
    r_rate = df["interest_rate"].corr(df["loan_paid_back"])
    print("Pearson r:", r_rate)

    print("Implied here is that there is a weak correlation between a lower interest rate resulting in a paid back loan.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Outlier Detection

    I will eliminate any outliers as determined by Tukey's Fences
    """)
    return


@app.cell
def _():
    #value = df.annual_income

    # Calculate quartiles
    #Q1 = value.quantile(0.25)
    #Q3 = value.quantile(0.75)
    #IQR = Q3 - Q1

    # Calculate Tukey's fences
    #lower_bound = Q1 - 1.5 * IQR
    #upper_bound = Q3 + 1.5 * IQR

    #print(f"Lower Bound (Tukey's Fence): {lower_bound}")
    #print(f"Upper Bound (Tukey's Fence): {upper_bound}")

    # Identify outliers using Tukey's fences
    #tukey_outliers = df[(value < lower_bound) | (value > upper_bound)]

    #print("\nOutliers (Tukey's Fences):")
    #print(tukey_outliers)
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
    c_df = df #Using a new dataframe, cleaned df, for processing. I wish to archive the original data.
    c_df
    return (c_df,)


@app.cell
def _(df):
    df["grade_subgrade"].unique()
    return


@app.cell
def _(pd):
    pd.set_option('future.no_silent_downcasting', True)
    return


@app.cell
def _(c_df):
    c_df["gender"] = c_df["gender"].replace({
        "Male": 0,
        "Female": 1,
        "Other": 2
    })

    c_df['marital_status'] = c_df["marital_status"].replace({
        "Single": 0,
        "Married": 1,
        "Divorced": 2,
        "Widowed": 3
    })

    c_df['education_level'] = c_df["education_level"].replace({
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3,
        "Other":4
    })
    c_df['employment_status'] = c_df["employment_status"].replace({
        "Unemployed": 0,
        "Employed": 1,
        "Self-employed": 2,
        "Retired": 3,
        "Student":4
    })
    c_df['loan_purpose'] = c_df["loan_purpose"].replace({
        "Debt consolidation": 0,
        "Home": 1,
        "Education": 2,
        "Vacation": 3,
        "Car": 4,
        "Medical": 5,
        "Business": 6,   
        "Other": 7,
    })
    c_df["grade_subgrade"] = c_df["grade_subgrade"].replace({
        "F1": 0,
        "F2": 0,
        "F3": 0,
        "F4": 0,
        "F5": 0,
        "E1": 1,
        "E2": 1,
        "E3": 1,
        "E4": 1,
        "E5": 1,
        "D1": 2,
        "D2": 2,
        "D3": 2,
        "D4": 2,
        "D5": 2,
        "C1": 3,
        "C2": 3,
        "C3": 3,
        "C4": 3,
        "C5": 3,
        "B1": 4,
        "B2": 4,
        "B3": 4,
        "B4": 4,
        "B5": 4,
        "A1": 5,
        "A2": 5,
        "A3": 5,
        "A4": 5,
        "A5": 5
    })
    return


@app.cell
def _(c_df):
    #The post-processes database now contains no strings, and all numerical values.

    c_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    Initially, I believed that there was not much difference in grouping the subgrades together against leaving them separate. Later, this will be tested out through the first logistic regression test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modeling
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Test Train Data Split
    """)
    return


@app.cell
def _(c_df, train_test_split):
    train_df, test_df = train_test_split(c_df, test_size=0.2, random_state=42)
    X_train = train_df.drop(columns=["loan_paid_back"])
    Y_train = train_df["loan_paid_back"]

    X_test = test_df.drop(columns=["loan_paid_back"])
    Y_test = test_df["loan_paid_back"]
    return X_test, X_train, Y_test, Y_train


@app.cell
def _(mo):
    mo.md(r"""
    ### Logistic Regression
    """)
    return


@app.cell
def _(LogisticRegression, X_train, Y_train, metrics):
    #I will now employ one of the first and simplest models for the project -- a logistic regression.abs

    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()

    # fit the model with data
    model = logreg.fit(X_train, Y_train)

    y_pred = model.predict(X_train)

    #Confusion Matrix for the Logistic Regression
    logregCnf_matrix = metrics.confusion_matrix(Y_train, y_pred)
    logregScore = metrics.accuracy_score(Y_train, y_pred)
    print(logregCnf_matrix)
    print(logregScore)

    print("Using a simple logistic regression with default settings, we achieve an", logregScore ,"accuracy using our training data.")
    return logreg, model


@app.cell
def _(X_test, Y_test, metrics, model):
    # Using the Test Set

    y_pred_Test = model.predict(X_test)

    #Confusion Matrix for the Logistic Regression
    logregCnf_matrix_Test = metrics.confusion_matrix(Y_test, y_pred_Test)
    logregScore_Test = metrics.accuracy_score(Y_test, y_pred_Test)
    print(logregCnf_matrix_Test)
    print(logregScore_Test)

    print("Using a simple logistic regression with default settings, we achieve an", logregScore_Test, " accuracy using our training data.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### New Test for Checking if there is a viable difference between each separating the subgrades or not.
    """)
    return


@app.cell
def _(path, pd):
    #Defining Test Dataframe and Processing it.
    c2_df = pd.read_csv(path)

    c2_df["gender"] = c2_df["gender"].replace({
        "Male": 0,
        "Female": 1,
        "Other": 3
    })

    c2_df['marital_status'] = c2_df["marital_status"].replace({
        "Single": 0,
        "Married": 1,
        "Divorced": 2,
        "Widowed": 3
    })

    c2_df['education_level'] = c2_df["education_level"].replace({
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3,
        "Other":4
    })
    c2_df['employment_status'] = c2_df["employment_status"].replace({
        "Unemployed": 0,
        "Employed": 1,
        "Self-employed": 2,
        "Retired": 3,
        "Student":4
    })
    c2_df['loan_purpose'] = c2_df["loan_purpose"].replace({
        "Debt consolidation": 0,
        "Home": 1,
        "Education": 2,
        "Vacation": 3,
        "Car": 4,
        "Medical": 5,
        "Business": 6,   
        "Other": 7,
    })
    c2_df["grade_subgrade"] = c2_df["grade_subgrade"].replace({
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

    c2_df
    return (c2_df,)


@app.cell
def _(c2_df, train_test_split):
    train_df2, test_df2 = train_test_split(c2_df, test_size=0.2, random_state=42)
    X_train2 = train_df2.drop(columns=["loan_paid_back"])
    Y_train2 = train_df2["loan_paid_back"]

    X_test2 = test_df2.drop(columns=["loan_paid_back"])
    Y_test2 = test_df2["loan_paid_back"]
    return X_test2, X_train2, Y_test2, Y_train2


@app.cell
def _(X_train2, Y_train2, logreg, metrics):

    # fit the model with dat
    model2 = logreg.fit(X_train2, Y_train2)

    y_pred2 = model2.predict(X_train2)

    #Confusion Matrix for the Logistic Regression
    logregCnf_matrix2 = metrics.confusion_matrix(Y_train2, y_pred2)
    logregScore2 = metrics.accuracy_score(Y_train2, y_pred2)
    print(logregCnf_matrix2)
    print(logregScore2)

    print("Using a simple logistic regression with default settings, we achieve an", logregScore2 ,"accuracy using our training data.")
    return (model2,)


@app.cell
def _(X_test2, Y_test2, metrics, model2):
    y_pred2_test = model2.predict(X_test2)

    #Confusion Matrix for the Logistic Regression
    logregCnf_matrix2_test = metrics.confusion_matrix(Y_test2, y_pred2_test)
    logregScore2_test = metrics.accuracy_score(Y_test2, y_pred2_test)
    print(logregCnf_matrix2_test)
    print(logregScore2_test)

    print("Using a simple logistic regression with default settings, we achieve an", logregScore2_test ," accuracy using our training data.")
    return


@app.cell
def _():
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
