import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get data 
trainDataClean = pd.read_csv('../data/train_clean.csv')

# ------------------------------
# log transform the cleaned data
# ------------------------------ 
trainDataClean["log_annual_income"] = np.log1p(trainDataClean["annual_income"])

# save to a new CSV file
trainDataClean.to_csv("../data/train_clean_logged.csv", index=False)

# plot 
trainDataClean["log_annual_income"].hist(bins=50)
plt.title("Log(Annual Income) Distribution (Cleaned)")
plt.xlabel("log_annual_income")
plt.ylabel("Frequency")
plt.savefig("../images/log_annual_income_histogram.png")
plt.show()

# ------------------------
# Evaluation of tranformed cleaned data
# 
# Applying a log transformation produced a much more symmetric distribution,
# reducing the influence of very large income values and making the data
# closer to a normal shape. 
# ------------------------

# --------
# skewness
# --------

# skewness of annual income
skew_value = trainDataClean["log_annual_income"].skew()
print("Skewness:", skew_value)
