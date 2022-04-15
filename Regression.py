'''

The given script tests three kinds of regressors over a set of data after splitting it into training and testing data
**********
Linear regression performs better than other two algorithms if looked over an average
**********
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

print("--------------------Linear Regression--------------------")
traffic = pd.read_csv('traffic_behavior.csv')
features = traffic.drop(['Slowness in traffic (%)'],axis=1)
labels = traffic['Slowness in traffic (%)']
f_train,f_test,l_train,l_test = train_test_split(features,labels)
print("Size of training set:",f_train.shape[0],", Number of features:",f_train.shape[1])
print("Size of testing set:",f_test.shape[0],", Number of features:",f_train.shape[1])
lin_reg = LinearRegression()
lin_reg = lin_reg.fit(f_train,l_train)
l_pred = lin_reg.predict(f_test)
error = (l_test-l_pred)**2
rss = sum(error)
print("RSS error(e): ",rss)
print("Correlation(r) Weight for each feature: \n",lin_reg.coef_)
print("Intercept: ",lin_reg.intercept_)

print("--------------------Decision Tree Regression--------------------")
dec_tree = DecisionTreeRegressor(random_state=3,max_depth=10) 
dec_tree = dec_tree.fit(f_train,l_train)
l_pred = dec_tree.predict(f_test)
error = (l_test-l_pred)**2
rss = sum(error)
print("RSS error(e): ",rss)
print("Correlation(r) Weight for each feature: \n",dec_tree.feature_importances_)
print("Correlation(r): ",dec_tree.score(f_test,l_test))

print("--------------------MLP Regression--------------------")
mlp_reg = MLPRegressor(early_stopping=True) 
mlp_reg = mlp_reg.fit(f_train,l_train)
l_pred = mlp_reg.predict(f_test)
error = (l_test-l_pred)**2
rss = sum(error)
print("RSS error(e): ",rss)
print("Correlation(r) Weight for each feature: \n",(mlp_reg.coefs_))
print("Correlation(r): ",mlp_reg.score(f_test,l_test))


# In[ ]:




