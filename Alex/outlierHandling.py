import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# get data 
trainData = pd.read_csv('../train.csv')

numeric_cols = trainData.select_dtypes(include="number")

# -------------------------
# z-score outlier detection
# -------------------------
print('Z-SCORE OUTLIER DETECTION:')

# calculate z-scores
z_scores = numeric_cols.apply(zscore)

# Set a threshold
threshold = 5
outliers_z = (abs(z_scores) > threshold)

# Get rows with any outlier
outlier_rows_z = trainData[outliers_z.any(axis=1)]
print(outlier_rows_z)

# Outlier count per column
outlier_counts = outliers_z.sum().sort_values(ascending=False)
print("Outliers per column:")
print(outlier_counts)

# -------------------------
# Explanation of z-score testing 
# 
# I am using the z-score outlier detection since the dataset is so large
# at almost 60,000 rows of data. Choosing a threshold of 5 allows the most 
# extreme outliers in our data to be caught. 
# ------------------------


# -----------------------------
# removing the extreme outliers
# -----------------------------

# rows that have at least one outlier
rows_with_outlier = outliers_z.any(axis=1)

# remove them
trainData_clean = trainData[~rows_with_outlier].copy()

#trainData_clean.to_csv("train_clean.csv", index=False)
trainData_clean = pd.read_csv("./train_clean.csv")

print("CLEANED DATA SET:", trainData_clean)
# ------------------------
# Evaluation of cleaned data
# 
# The graph was still right skewed but not as much after removing the outliers. 
# With the graph being right skewed, suggests log transformation which will 
# compresses large values and spreads small values. 
# ------------------------

# ------------------------------
# log transform the cleaned data
# ------------------------------ 
trainData_clean["log_annual_income"] = np.log1p(trainData_clean["annual_income"])

# plot 
trainData_clean["log_annual_income"].hist(bins=50)
plt.title("Log(Annual Income) Distribution (Cleaned)")
plt.xlabel("log_annual_income")
plt.ylabel("Frequency")
plt.savefig("log_annual_income_histogram.png")
plt.show()

# ------------------------
# Evaluation of tranformed cleaned data
# 
# Applying a log transformation produced a much more symmetric distribution,
# reducing the influence of very large income values and making the data
# closer to a normal shape. 
# ------------------------