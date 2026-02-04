import pandas as pd

# get data 
trainData = pd.read_csv('../train.csv')

# print first 5 rows
print('Head:', trainData.head())

# dimensions
#print(trainData.dim())

# index
print('Information:', trainData.info)