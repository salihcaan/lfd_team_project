# libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)


def test():
    # read file
    df = pd.read_csv('test.csv')
    df.drop(['X3', 'X31', 'X32', 'X127', 'X128', 'X590'], axis=1)

    # seperate label and features
    x = df.loc[:, df.columns[0:-1]].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # PCA
    pca = PCA(n_components=23)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents)
    return principalDf




# read file
df = pd.read_csv('train.csv')
df.drop(['X3', 'X31', 'X32', 'X127', 'X128', 'X590'], axis=1)

# seperate label and features
x = df.loc[:, df.columns[0:-1]].values
y = df.loc[:, df.columns[-1]].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

# PCA
pca = PCA(n_components=23)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)

finalDf = pd.concat([principalDf, df[['class']]], axis = 1)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LinearRegression()
model.fit(principalDf, y)
predicted_classes = model.predict(test())

import csv

with open('submission.csv', mode='w') as employee_file:
    submission = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    submission.writerow(['ID', 'Predicted'])
    a = 1
    for i in predicted_classes:
        submission.writerow([str(a), i])
        a = a + 1

#accuracy = accuracy_score(y.flatten(), predicted_classes)
#print(accuracy)
#parameters = model.coef


grand_truth_label = [0,0,0,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,
                     1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0]

accuracy = 0

for i in range(80):
    if i not in [1, 2, 4, 7, 10, 13, 14, 21, 22, 24, 26, 28, 29, 32, 33, 38, 39, 40, 42, 44, 46, 47
          , 50, 51, 53, 54, 56, 59, 61, 63, 64, 67, 69, 71, 72, 73, 75, 77, 78, 79]:
        accuracy += grand_truth_label[i] == round(predicted_classes[i])


print(accuracy/40 * 100)