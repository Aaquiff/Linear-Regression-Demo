# Starting Script
print("Starting Script")

import numpy as np
from sklearn import datasets, linear_model

# Function to read file into array
def readMyFile(filename):
    import csv
    data = []
 
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            data.append(row)
 
    return data
 
# Linear Regression TODO
 
data = readMyFile('insurance.csv')

dataTrain = data[:-20]
dataTest = data[:-20]

a = np.array(data)

b = a.data[0:2]

print(b)
