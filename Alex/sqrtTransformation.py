import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get data 
trainDataClean = pd.read_csv('../data/train_clean.csv')

# ------------------------------
# square root transform the cleaned data
# ------------------------------ 
trainDataClean["sqrt_annual_income"] = np.sqrt(trainDataClean["annual_income"])

# plot 
trainDataClean["sqrt_annual_income"].hist(bins=50)
plt.title("Square Root(Annual Income) Distribution (Cleaned)")
plt.xlabel("sqrt_annual_income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../images/sqrt_annual_income_histogram.png")
plt.show()

# ------------------------
# Evaluation of transformed cleaned data
# 
# Applying a square root transformation reduced skewness in the income
# distribution by compressing very large values. While it does not
# normalize the data as strongly as a log transformation, it makes
# the distribution more symmetric and less influenced by extreme outliers.
# ------------------------
