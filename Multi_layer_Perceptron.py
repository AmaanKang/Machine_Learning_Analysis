'''
The code uses MLP classifier to test five different datasets
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
n = 1
while n < 6:
    print()
    print("************** Dataset ",n," ****************")
    if(n == 5):
        print("File: weight_lifting.csv")
        data = pd.read_csv('weight_lifting.csv')
        data = data.replace('#DIV/0!',0)
        LE = LabelEncoder()
        data['classe'] = LE.fit_transform(data['classe'])
    else:
        nStr = str(n)
        print('File: 822179_'+nStr+'.csv')
        data = pd.read_csv('822179_'+nStr+'.csv',header=None)
    data = data.fillna(0)
    nCol = len(data.columns)-1
    features = data.iloc[:,0:nCol]
    label = data.iloc[:,nCol]
    X_train, X_test, y_train, y_test = train_test_split(features, label)
    
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Decision Tree: Accuracy - ",round(score*100,2),"%")
    
    mlp_clf = MLPClassifier(max_iter=300,random_state=3,n_iter_no_change=5,activation="logistic",tol=0.001)
    mlp_clf.fit(X_train,y_train) 
    y_pred=mlp_clf.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    n += 1
    print("MLP: Accuracy - ",round(score*100,2),"%"," Number of iterations- ",mlp_clf.n_iter_)
    print("MLP: max_iter=300,random_state=3,n_iter_no_change=5,activation='logistic',tol=0.001")


