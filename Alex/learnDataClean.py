import pandas as pd

# get data 
trainDataClean = pd.read_csv('./train_clean.csv')

# print first 5 rows
print('Head:', trainDataClean.head())

# index
print('Information:', trainDataClean.info)
