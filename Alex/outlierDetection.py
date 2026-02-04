import pandas as pd
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

# Set a threshold (common: 3)
threshold = 3
outliers_z = (abs(z_scores) > threshold)

# Get rows with any outlier
outlier_rows_z = trainData[outliers_z.any(axis=1)]
print(outlier_rows_z)

# Outlier count per column
outlier_counts = outliers_z.sum().sort_values(ascending=False)
print("Outliers per column:")
print(outlier_counts)

# -----------------------
# tukey outlier detection
# -----------------------
print('TUKEY OUTLIER DECTECTION')

Q1 = numeric_cols.quantile(0.25, numeric_only=True)
Q3 = numeric_cols.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1

# Tukey's fences: 1.5*IQR for mild, 3*IQR for extreme outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_tukey = ((numeric_cols < lower_bound) | (numeric_cols > upper_bound))
outlier_rows_tukey = numeric_cols[outliers_tukey.any(axis=1)]
print(outlier_rows_tukey)