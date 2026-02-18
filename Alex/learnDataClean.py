import pandas as pd
import matplotlib.pyplot as plt

# get data 
trainDataClean = pd.read_csv('./train_clean.csv')

# print first 5 rows
print('Head:', trainDataClean.head())

# index
print('Information:', trainDataClean.info)

# plot vs annuanl income
trainDataClean["annual_income"].hist(bins=50)
plt.title("Annual Income Distribution (Cleaned)")
plt.xlabel("annual_income")
plt.ylabel("Frequency")
plt.savefig("trainDataCleaned.png")
plt.show()
