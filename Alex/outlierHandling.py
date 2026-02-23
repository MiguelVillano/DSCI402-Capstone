import pandas as pd
import numpy as np
import os
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
trainDataClean = trainData[~rows_with_outlier].copy()

file_path = "train_clean.csv"

if not os.path.exists(file_path):
    # File does NOT exist → create it
    trainDataClean.to_csv(file_path, index=False)
    trainDataClean = trainDataClean.copy()
else:
    # File exists → read it
    trainDataClean = pd.read_csv(file_path)

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
trainDataClean["log_annual_income"] = np.log1p(trainDataClean["annual_income"])

# plot 
trainDataClean["log_annual_income"].hist(bins=50)
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