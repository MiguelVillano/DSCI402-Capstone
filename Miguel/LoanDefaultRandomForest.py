import marimo

__generated_with = "0.23.1"
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score
    )
    from sklearn.inspection import permutation_importance
    from sklearn.tree import plot_tree
    from sklearn.model_selection import train_test_split

    return (
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        os,
        pd,
        plt,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    return PCA, StandardScaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset Selection
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
    dataset_selector = mo.ui.dropdown( options=dataset_files, value=dataset_files[0] if dataset_files else None, label="Select Dataset" )
    dataset_selector
    return (dataset_selector,)


@app.cell
def _(DATA_FOLDER, dataset_selector, os, pd):
    selected_path = os.path.join(DATA_FOLDER, dataset_selector.value)

    df = pd.read_csv(selected_path)

    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Preprocessing
    """)
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
    return


@app.cell
def _(df, train_test_split):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(columns=["loan_paid_back","annual_income", "id", "grade_subgrade"])
    y_train = train_df["loan_paid_back"]

    X_test = test_df.drop(columns=["loan_paid_back","annual_income", "id", "grade_subgrade"])
    y_test = test_df["loan_paid_back"]
    return X_test, X_train, test_df, train_df, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Forest
    """)
    return


@app.cell
def _(RandomForestClassifier):
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    return (rf_model,)


@app.cell
def _(X_test, X_train, rf_model, y_train):
    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    return rf_preds, rf_probs


@app.cell
def _(
    accuracy_score,
    classification_report,
    confusion_matrix,
    rf_preds,
    rf_probs,
    roc_auc_score,
    y_test,
):
    print("Accuracy:", accuracy_score(y_test, rf_preds))
    print("ROC AUC:", roc_auc_score(y_test, rf_probs))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))

    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coefficient Investigation
    """)
    return


@app.cell
def _(X_train, pd, rf_model):
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importances:")
    print(importance_df)
    return (importance_df,)


@app.cell
def _(importance_df, plt):
    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )
    plt.gca().invert_yaxis()

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multicollinearity Investigation
    """)
    return


@app.cell
def _(X_train, pd):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = X_train.drop(columns=["gender","marital_status","education_level","employment_status","loan_purpose","grade_subgrade","id"])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    print(vif_data)
    return (variance_inflation_factor,)


@app.cell
def _(test_df, train_df):
    X_train2 = train_df[["credit_score","yeojohnson_annual_income","interest_rate"]]
    y_train2 = train_df["loan_paid_back"]

    X_test2 = test_df[["credit_score","yeojohnson_annual_income","interest_rate"]]
    y_test2 = test_df["loan_paid_back"]
    return X_test2, X_train2, y_test2, y_train2


@app.cell
def _(X_test2, X_train2, rf_model, y_train2):
    rf_model.fit(X_train2, y_train2)

    rf_preds2 = rf_model.predict(X_test2)
    rf_probs2 = rf_model.predict_proba(X_test2)[:, 1]
    return rf_preds2, rf_probs2


@app.cell
def _(
    accuracy_score,
    classification_report,
    confusion_matrix,
    rf_preds2,
    rf_probs2,
    roc_auc_score,
    y_test2,
):
    print("Accuracy:", accuracy_score(y_test2, rf_preds2))
    print("ROC AUC:", roc_auc_score(y_test2, rf_probs2))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test2, rf_preds2))

    print("\nClassification Report:")
    print(classification_report(y_test2, rf_preds2))
    return


@app.cell
def _(X_train2, pd, rf_model):
    importance_df2 = pd.DataFrame({
        "Feature": X_train2.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importances:")
    print(importance_df2)
    return


@app.cell
def _(X_train, pd, variance_inflation_factor):
    X_vif2 = X_train[["debt_to_income_ratio","yeojohnson_annual_income","interest_rate","credit_score"]]
    vif_data2 = pd.DataFrame()
    vif_data2["Feature"] = X_vif2.columns
    vif_data2["VIF"] = [variance_inflation_factor(X_vif2.values, i) for i in range(X_vif2.shape[1])]

    print(vif_data2)
    return


@app.cell
def _():
    # Do PCA on Ann_Income, Interest Rate, Credit Score into Financials
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Principal Component Analysis
    """)
    return


@app.cell
def _(PCA, StandardScaler, df, pd):
    scaler = StandardScaler()
    pca_FI = pd.DataFrame
    pca_FI = df[["yeojohnson_annual_income","interest_rate","credit_score"]]
    pcaModel_FI = PCA(n_components=1)
    pca_FI_scaled = scaler.fit_transform(pca_FI)
    FI_pca = pcaModel_FI.fit_transform(pca_FI_scaled)
    return (FI_pca,)


@app.cell
def _(FI_pca, df):
    df["Financials"] = FI_pca
    return


@app.cell
def _(df, train_test_split):
    train_df3, test_df3 = train_test_split(df, test_size=0.2, random_state=42)
    X_train3 = train_df3.drop(columns=["loan_paid_back","id","annual_income","grade_subgrade","yeojohnson_annual_income","interest_rate","credit_score"])
    y_train3 = train_df3["loan_paid_back"]

    X_test3 = test_df3.drop(columns=["loan_paid_back","id","annual_income","grade_subgrade","yeojohnson_annual_income","interest_rate","credit_score"])
    y_test3 = test_df3["loan_paid_back"]
    return X_test3, X_train3, test_df3, train_df3, y_test3, y_train3


@app.cell
def _(X_test3, X_train3, rf_model, y_train3):
    rf_model.fit(X_train3, y_train3)

    rf_preds3 = rf_model.predict(X_test3)
    rf_probs3 = rf_model.predict_proba(X_test3)[:, 1]
    return rf_preds3, rf_probs3


@app.cell
def _(
    accuracy_score,
    classification_report,
    confusion_matrix,
    rf_preds3,
    rf_probs3,
    roc_auc_score,
    y_test3,
):
    print("Accuracy:", accuracy_score(y_test3, rf_preds3))
    print("ROC AUC:", roc_auc_score(y_test3, rf_probs3))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test3, rf_preds3))

    print("\nClassification Report:")
    print(classification_report(y_test3, rf_preds3))
    return


@app.cell
def _(X_train3, pd, plt, rf_model):
    importance_df3 = pd.DataFrame({
        "Feature": X_train3.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df3["Feature"],
        importance_df3["Importance"]
    )
    plt.gca().invert_yaxis()

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.show()
    return


@app.cell
def _(X_train3, pd, variance_inflation_factor):
    X_vif3 = X_train3[["Financials","debt_to_income_ratio","loan_amount"]]
    vif_data3 = pd.DataFrame()
    vif_data3["Feature"] = X_vif3.columns
    vif_data3["VIF"] = [variance_inflation_factor(X_vif3.values, i) for i in range(X_vif3.shape[1])]

    print(vif_data3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fourth Test
    """)
    return


@app.cell
def _(test_df3, train_df3):
    X_train4 = train_df3.drop(columns=["loan_paid_back","id","annual_income","grade_subgrade","yeojohnson_annual_income","interest_rate","credit_score","employment_status"])
    y_train4 = train_df3["loan_paid_back"]

    X_test4 = test_df3.drop(columns=["loan_paid_back","id","annual_income","grade_subgrade","yeojohnson_annual_income","interest_rate","credit_score","employment_status"])
    y_test4 = test_df3["loan_paid_back"]
    return X_test4, X_train4, y_test4, y_train4


@app.cell
def _(X_test4, X_train4, rf_model, y_train4):
    rf_model.fit(X_train4, y_train4)

    rf_preds4 = rf_model.predict(X_test4)
    rf_probs4 = rf_model.predict_proba(X_test4)[:, 1]
    return rf_preds4, rf_probs4


@app.cell
def _(
    accuracy_score,
    classification_report,
    confusion_matrix,
    rf_preds4,
    rf_probs4,
    roc_auc_score,
    y_test4,
):
    print("Accuracy:", accuracy_score(y_test4, rf_preds4))
    print("ROC AUC:", roc_auc_score(y_test4, rf_probs4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test4, rf_preds4))

    print("\nClassification Report:")
    print(classification_report(y_test4, rf_preds4))
    return


@app.cell
def _(X_train4, pd, plt, rf_model):
    importance_df4 = pd.DataFrame({
        "Feature": X_train4.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df4["Feature"],
        importance_df4["Importance"]
    )
    plt.gca().invert_yaxis()

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
