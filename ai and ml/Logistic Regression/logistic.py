# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:41:54 2022

@author: vineela
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

df = pd.read_csv("D:/machine learning lab/py-master_codebasics/py-master/ML/7_logistic_reg/insurance_data.csv")
df.head()

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)

X_test

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


model.fit(X_train, y_train)
X_test

y_predicted = model.predict(X_test)
model.predict_proba(X_test)

print(metrics.confusion_matrix(y_predicted, y_test))
y_predicted

X_test



# model.coef_
# model.intercept_

# import math
# def sigmoid(x):
#   return 1 / (1 + math.exp(-x))


# def prediction_function(age):
#     z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
#     y = sigmoid(z)
#     return y

# age = 35
# prediction_function(age)

# age = 43
# prediction_function(age)