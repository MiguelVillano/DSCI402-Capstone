import pandas as pd
import matplotlib.pyplot as plt

# get data 
trainData = pd.read_csv('../train.csv')

# print first 5 rows
print('Head:', trainData.head())

# index
print('Information:', trainData.info)

# plot vs annual_income
trainData["annual_income"].hist(bins=50)
plt.title("Annual Income Distribution")
plt.xlabel("annual_income")
plt.ylabel("Frequency")
plt.savefig("trainData.png")
plt.show()