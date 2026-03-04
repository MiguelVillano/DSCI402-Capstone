import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

# get data 
trainDataClean = pd.read_csv('../data/train_clean.csv')

# ------------------------------
# Apply Yeo-Johnson transformation
# ------------------------------
pt = PowerTransformer(method='yeo-johnson', standardize=False)  # don't standardize if you want raw transformed values
trainDataClean["yeojohnson_annual_income"] = pt.fit_transform(trainDataClean[["annual_income"]])

# print lambda(s)
fitted_lambda = pt.lambdas_[0]
print(f"Optimal lambda for Yeo-Johnson: {fitted_lambda:.4f}")

# save to a new CSV file
trainDataClean.to_csv("../data/train_clean_yeojohnson.csv", index=False)

# plot 
trainDataClean["yeojohnson_annual_income"].hist(bins=50)
plt.title("Yeo-Johnson(Annual Income) Distribution (Cleaned)")
plt.xlabel("yeojohnson_annual_income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../images/yeojohnson_annual_income_histogram.png")
plt.show()

# ------------------------
# Evaluation of Yeo-Johnson transformed data
# 
# The Yeo-Johnson transformation automatically finds a power (lambda) to make
# the distribution more symmetric and closer to normal. Unlike Box-Cox, it can
# handle zero and negative values directly. This makes it a very flexible 
# option for skewed or non-positive income data.
# ------------------------

# --------
# skewness
# --------

# skewness of annual income
skew_value = trainDataClean["yeojohnson_annual_income"].skew()
print("Skewness:", skew_value)
