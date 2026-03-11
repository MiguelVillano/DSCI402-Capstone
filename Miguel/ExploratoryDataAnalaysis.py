import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialization
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = 'train.csv'
    c_df = pd.read_csv(path)
    return c_df, mo, path, pd, plt, sns


@app.cell
def _(c_df):
    c_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining New Dataframes
    """)
    return


@app.cell
def _(path, pd):
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
def _(pd):
    path2 = '../data/train_clean_logged.csv'
    logged_df = pd.read_csv(path2)
    return (logged_df,)


@app.cell
def _(logged_df):
    logged_df
    return


@app.cell
def _(logged_df):
    small_df = logged_df.sample(n=50000, random_state=42)
    small_df
    return (small_df,)


@app.cell
def _(logged_df):
    smaller_df = logged_df.sample(n=10000, random_state=42)
    smaller_df
    return (smaller_df,)


@app.cell
def _(logged_df):
    smallest_df = logged_df.sample(n=5000, random_state=42)
    smallest_df
    return (smallest_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Sampling Investigation
    """)
    return


@app.cell
def _(c_df, sns):
    sns.violinplot( y=c_df["credit_score"], x=c_df["loan_paid_back"])
    return


@app.cell
def _(small_df, sns):
    sns.violinplot( y=small_df["credit_score"], x=small_df["loan_paid_back"])
    return


@app.cell
def _(smaller_df, sns):
    sns.violinplot( y=smaller_df["credit_score"], x=smaller_df["loan_paid_back"])
    return


@app.cell
def _(smallest_df, sns):
    sns.violinplot( y=smallest_df["credit_score"], x=smallest_df["loan_paid_back"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing Continuous Values
    """)
    return


@app.cell
def _(smallest_df, sns):
    sns.boxplot( y=smallest_df["loan_amount"], x=smallest_df["loan_paid_back"])
    return


@app.cell
def _(small_df, sns):
    sns.boxplot( y=small_df["debt_to_income_ratio"], x=small_df["loan_paid_back"])
    return


@app.cell
def _(smaller_df, sns):
    sns.boxplot( y=smaller_df["interest_rate"], x=smaller_df["loan_paid_back"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing Categorical Values
    """)
    return


@app.cell
def _(plt, small_df, sns):
    sns.pointplot(
        data=small_df,
        x="education_level",
        y="loan_paid_back",
        estimator="mean"
    )

    plt.ylabel("Paid Back Probability")
    plt.xticks(rotation=45)
    plt.show()
    return


@app.cell
def _(c_df):
    c_df["loan_paid_back"].value_counts(normalize=True)
    return


@app.cell
def _(plt, small_df, sns):
    sns.pointplot(
        data=small_df,
        x="loan_purpose",
        y="loan_paid_back",
        estimator="mean"
    )

    plt.ylabel("Paid Back Probability")
    plt.xticks(rotation=45)
    plt.show()
    return


@app.cell
def _(plt, small_df, sns):
    sns.pointplot(
        data=small_df,
        x="grade_subgrade",
        y="loan_paid_back",
        estimator="mean"
    )

    plt.ylabel("Paid Back Probability")
    plt.xticks(rotation=45)
    plt.show()
    return


@app.cell
def _(plt, small_df, sns):
    sns.pointplot(
        data=small_df,
        x="marital_status",
        y="loan_paid_back",
        estimator="mean"
    )

    plt.ylabel("Paid Back Probability")
    plt.xticks(rotation=45)
    plt.show()
    return


@app.cell
def _(plt, small_df, sns):
    sns.pointplot(
        data=small_df,
        x="employment_status",
        y="loan_paid_back",
        estimator="mean"
    )

    plt.ylabel("Paid Back Probability")
    plt.xticks(rotation=45)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handling Categorical Features
    """)
    return


@app.cell
def _(c2_df):
    c2_df
    return


@app.cell
def _(c2_df, pd):
    df_encoded_loans = pd.get_dummies(
        c2_df,
        columns=["loan_purpose"],
        drop_first=True
    )
    loan_purpose_cols = df_encoded_loans.filter(like="loan_purpose")
    return (loan_purpose_cols,)


@app.cell
def _(c2_df, pd):
    df_encoded_education = pd.get_dummies(
        c2_df,
        columns=["education_level"],
        drop_first=True
    )
    education_cols = df_encoded_education.filter(like="education_level")
    return (education_cols,)


@app.cell
def _(c2_df, pd):
    df_encoded_employment = pd.get_dummies(
        c2_df,
        columns=["employment_status"],
        drop_first=True
    )
    employment_cols = df_encoded_employment.filter(like="employment_status")
    return (employment_cols,)


@app.cell
def _():
    return


@app.cell
def _(c2_df, education_cols, employment_cols, loan_purpose_cols, pd):
    df_encoded = c2_df.drop(columns=["employment_status","education_level","loan_purpose","grade_subgrade","marital_status","gender"])
    df_encoded = pd.concat(
        [df_encoded, loan_purpose_cols, education_cols, employment_cols],
        axis=1
    )
    return (df_encoded,)


@app.cell
def _(df_encoded):
    df_encoded
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlation Investigation
    """)
    return


@app.cell
def _(df_encoded):
    income_to_loan_corr = df_encoded["annual_income"].corr(df_encoded["loan_amount"])
    income_to_loan_corr
    return


@app.cell
def _(df_encoded):
    bachelors_loan_corr = df_encoded["education_level_1"].corr(df_encoded["loan_purpose_2"])
    bachelors_loan_corr
    return


@app.cell
def _(df_encoded):
    masters_loan_corr = df_encoded["education_level_2"].corr(df_encoded["loan_purpose_2"])
    masters_loan_corr 
    return


@app.cell
def _(df_encoded):
    df_encoded.corr(method="pearson")["loan_paid_back"].sort_values(ascending=False)
    return


if __name__ == "__main__":
    app.run()
