import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox

# get data 
trainDataClean = pd.read_csv('../data/train_clean.csv')

# ------------------------------
# Ensure all values are positive for Box-Cox
# ------------------------------
# If there are zeros, add a small constant
income_shifted = trainDataClean["annual_income"] + 1e-6  

# Apply Box-Cox transformation
trainDataClean["boxcox_annual_income"], fitted_lambda = boxcox(income_shifted)

print(f"Optimal lambda for Box-Cox: {fitted_lambda:.4f}")

# plot 
trainDataClean["boxcox_annual_income"].hist(bins=50)
plt.title("Box-Cox(Annual Income) Distribution (Cleaned)")
plt.xlabel("boxcox_annual_income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../images/boxcox_annual_income_histogram.png")
plt.show()

# ------------------------
# Evaluation of Box-Cox transformed data
# 
# The Box-Cox transformation automatically selects the power (lambda) that
# makes the distribution as close to normal as possible. This reduces skewness
# more effectively than log or square root, while still preserving the relative
# differences between incomes.
# ------------------------
