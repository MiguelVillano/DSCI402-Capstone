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
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn import metrics



    return (
        LogisticRegression,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        os,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_curve, roc_auc_score


    return PCA, StandardScaler, roc_auc_score, roc_curve


@app.cell
def _():
    import statsmodels.api as sm

    return (sm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Selection
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Data Preprocessing
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
    X_train = train_df.drop(columns=["loan_paid_back","annual_income"])
    y_train = train_df["loan_paid_back"]

    X_test = test_df.drop(columns=["loan_paid_back","annual_income"])
    y_test = test_df["loan_paid_back"]
    return X_test, X_train, test_df, train_df, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistic Regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Initial Test
    """)
    return


@app.cell
def _(LogisticRegression, X_test, X_train, y_train):
    log_model = LogisticRegression(
        solver='lbfgs',
        max_iter=500,
        tol=1e-3,
        class_weight='balanced',
        random_state=42
    )

    log_model.fit(X_train, y_train)

    log_preds = log_model.predict(X_test)
    return log_model, log_preds


@app.cell
def _(accuracy_score, log_preds, y_test):
    log_acc = accuracy_score(y_test, log_preds)
    print("Logistic Regression Accuracy:", log_acc)
    return


@app.cell
def _(X_train):
    feature_names = X_train.columns
    feature_names
    return (feature_names,)


@app.cell
def _(feature_names, log_model, pd):
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": log_model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print(coef_df)
    return (coef_df,)


@app.cell
def _(coef_df, plt):
    plt.figure(figsize=(10,6))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"])
    plt.title("Logistic Regression Feature Coefficients")
    plt.axvline(0, color='black')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Second Test
    """)
    return


@app.cell
def _(test_df, train_df):
    #Since it is apparent that employment status and debt-to-income ratio have dominant weights over the rest of the features, we will go forward by removing it to investigate patterns amongst the lesser dominant features.

    X_train2 = train_df.drop(columns=["loan_paid_back","annual_income","employment_status","debt_to_income_ratio"])
    y_train2 = train_df["loan_paid_back"]

    X_test2 = test_df.drop(columns=["loan_paid_back","annual_income","employment_status","debt_to_income_ratio"])
    y_test2 = test_df["loan_paid_back"]
    return X_test2, X_train2, y_test2, y_train2


@app.cell
def _(X_test2, X_train2, log_model, y_train2):
    log_model.fit(X_train2, y_train2)

    log_preds2 = log_model.predict(X_test2)
    return (log_preds2,)


@app.cell
def _(accuracy_score, log_preds2, y_test2):
    log_acc2 = accuracy_score(y_test2, log_preds2)
    print("Logistic Regression Accuracy:", log_acc2)
    return


@app.cell
def _(X_train2, log_model, pd):
    feature_names2 = X_train2.columns

    coef_df2 = pd.DataFrame({
        "Feature": feature_names2,
        "Coefficient": log_model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print(coef_df2)
    return (coef_df2,)


@app.cell
def _(coef_df2, plt):
    plt.figure(figsize=(10,6))
    plt.barh(coef_df2["Feature"], coef_df2["Coefficient"])
    plt.title("Logistic Regression Feature Coefficients")
    plt.axvline(0, color='black')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assumptions Validation
    """)
    return


@app.cell
def _(X_train, pd):
    #While it is strange to check assumptions after two models, I believe it to be important as upon removing the dominant feature of employment status, the coefficients are very low. This may indicate that some assumptions are violated.abs
    # predicted log-odds

    #Multicollinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = X_train.drop(columns=["gender","marital_status","education_level","employment_status","loan_purpose","grade_subgrade","id"])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    print(vif_data)
    return variance_inflation_factor, vif_data


@app.cell
def _(vif_data):
    # This result is very bad. We wil calculate R2 of each feature in order to see how much variation in each feature is captured by others.


    vif_df = vif_data.copy()
    vif_df["R_squared"] = 1 - (1 / vif_df["VIF"])

    print(vif_df.sort_values("VIF", ascending=False))
    return


@app.cell
def _(X_train, pd, variance_inflation_factor):
    # Since the R2 are relatively similar, we will remove Annual Income since in the original Logistic Regression, it was the weakest feature. We will also remove credit_score since it is involved in Grade_Subgrade

    X_vif2 = X_train.drop(columns=["gender","marital_status","education_level","employment_status","loan_purpose","grade_subgrade","id","yeojohnson_annual_income","credit_score"])
    vif_data2 = pd.DataFrame()
    vif_data2["Feature"] = X_vif2.columns
    vif_data2["VIF"] = [variance_inflation_factor(X_vif2.values, i) for i in range(X_vif2.shape[1])]

    print(vif_data2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Third Test
    """)
    return


@app.cell
def _(test_df, train_df):
    #This test will eliminate the multicollinearity in the dataset, and hopefully result in a higher accuracy score.

    X_train3 = train_df.drop(columns=["loan_paid_back","annual_income","yeojohnson_annual_income","credit_score"])
    y_train3 = train_df["loan_paid_back"]

    X_test3 = test_df.drop(columns=["loan_paid_back","annual_income","yeojohnson_annual_income","credit_score"])
    y_test3 = test_df["loan_paid_back"]
    return X_test3, X_train3, y_test3, y_train3


@app.cell
def _(X_test3, X_train3, log_model, y_train3):
    log_model.fit(X_train3, y_train3)

    log_preds3 = log_model.predict(X_test3)
    return (log_preds3,)


@app.cell
def _(accuracy_score, log_preds3, y_test3):
    log_acc3 = accuracy_score(y_test3, log_preds3)
    print("Logistic Regression Accuracy:", log_acc3)

    return


@app.cell
def _(confusion_matrix, log_preds3, y_test3):
    print("Confusion Matrix: ", confusion_matrix(y_test3, log_preds3))
    return


@app.cell
def _(classification_report, log_preds3, y_test3):
    print("Classification Report: ", classification_report(y_test3, log_preds3))
    return


@app.cell
def _(X_train3, log_model, pd):
    feature_names3 = X_train3.columns

    coef_df3 = pd.DataFrame({
        "Feature": feature_names3,
        "Coefficient": log_model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print(coef_df3)
    return (coef_df3,)


@app.cell
def _(coef_df3, plt):
    plt.figure(figsize=(10,6))
    plt.barh(coef_df3["Feature"], coef_df3["Coefficient"])
    plt.title("Logistic Regression Feature Coefficients")
    plt.axvline(0, color='black')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Principal Component Analysis
    """)
    return


@app.cell
def _(LogisticRegression):
    log_model = LogisticRegression(
        solver='lbfgs',
        max_iter=500,
        tol=1e-3,
        class_weight='balanced',
        random_state=42
    )
    return (log_model,)


@app.cell
def _(StandardScaler, X_test, X_train):
    #This fourth test will apply PCA to fully eliminate the Multicollinearity still plaguing the dataset. Tests show that removing dominant features simply make way for another feature to become dominant, indicating that knowledge in the dataset is covered by all datasets.

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled


@app.cell
def _(PCA, X_test_scaled, X_train_scaled):
    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("Number of components:", pca.n_components_)
    return X_test_pca, X_train_pca


@app.cell
def _(test_df, train_df):
    y_train4 = train_df["loan_paid_back"]
    y_test4 = test_df["loan_paid_back"]
    return y_test4, y_train4


@app.cell
def _(X_test_pca, X_train_pca, log_model, y_train4):
    log_model.fit(X_train_pca, y_train4)

    log_preds4 = log_model.predict(X_test_pca)
    return (log_preds4,)


@app.cell
def _(accuracy_score, log_preds4, y_test4):
    log_acc4 = accuracy_score(y_test4, log_preds4)
    print("Logistic Regression Accuracy:", log_acc4)
    return


@app.cell
def _(confusion_matrix, log_preds3, log_preds4, y_test3, y_test4):
    print("Old Confusion Matrix: ", confusion_matrix(y_test3, log_preds3))
    print("New Confusion Matrix: ", confusion_matrix(y_test4, log_preds4))

    #Here it can be seen that the model with less dominant features makes more correct predictions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Threshold Lowering
    """)
    return


@app.cell
def _(X_test_pca, log_model):
    #Based on the model with no multicollinearity, the threshold needs to be lowered since too many false negatives (Good Loaners being Denied)

    threshold = 0.30  # example
    #test3_approved_prob = log_model.predict_proba(test3_df)[:, 1]
    #test3_custom_appPreds = (test3_approved_prob >= threshold).astype(int)

    pca_approved_prob = log_model.predict_proba(X_test_pca)[:, 1]
    pca_custom_appPreds = (pca_approved_prob >= threshold).astype(int)
    return (pca_custom_appPreds,)


@app.cell
def _(
    confusion_matrix,
    log_preds3,
    log_preds4,
    pca_custom_appPreds,
    y_test3,
    y_test4,
):
    print("Test 3: ", confusion_matrix(y_test3, log_preds3))
    #print("Lowered Threshold Test 3: ", confusion_matrix(Y_test3_df, test3_custom_appPreds))

    print("PCA: ", confusion_matrix(y_test4, log_preds4))
    print("Lowered Threshold PCA: ", confusion_matrix(y_test4, pca_custom_appPreds))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ROC Curve
    """)
    return


@app.cell
def _():
    #X_train3 = train_df.drop(columns=["loan_paid_back","annual_income","yeojohnson_annual_income","credit_score"])
    #y_train3 = train_df["loan_paid_back"]

    #X_test3 = test_df.drop(columns=["loan_paid_back","annual_income","yeojohnson_annual_income","credit_score"])
    #y_test3 = test_df["loan_paid_back"]
    return


@app.cell
def _(log_preds3, train_df, y_test3):
    y_test3, log_preds3

    test3_df = train_df.drop(columns=["annual_income","yeojohnson_annual_income","credit_score"])
    y_test3_df = train_df["loan_paid_back"]
    return


@app.cell
def _(
    X_test3,
    X_train3,
    log_model,
    plt,
    roc_auc_score,
    roc_curve,
    y_test3,
    y_train3,
):
    # train
    log_model.fit(X_train3, y_train3)

    # predict
    probs = log_model.predict_proba(X_test3)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test3, probs)
    auc = roc_auc_score(y_test3, probs)

    print("AUC:", auc)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P values
    """)
    return


@app.cell
def _(X_train, pd):
    X_train3_cleaned = X_train.apply(pd.to_numeric, errors='coerce')
    return (X_train3_cleaned,)


@app.cell
def _(X_train3_cleaned, sm, y_train):
    X_sm = sm.add_constant(X_train3_cleaned.drop(columns=["id"]))  # adds intercept
    model = sm.Logit(y_train, X_sm)
    result = model.fit()

    print(result.summary())
    return


if __name__ == "__main__":
    app.run()
