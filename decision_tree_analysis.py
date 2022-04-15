'''
This part of the code tests the accuracy of decision tree over different versions of the classifier
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
frogs_data = pd.read_csv('frogs.csv')
feature_name = frogs_data.columns[0:14]
features = frogs_data.drop(['Green frogs'],axis=1)
label = frogs_data['Green frogs']
run = 0
crit_gini = 0
crit_entropy = 0
max_dep_10 = 0
max_dep_3 = 0
min_split_3 = 0
min_split_10 = 0
max_f_auto = 0
max_f_sqrt = 0
while run < 50:
    if run < 5:
        print(run)
    features_train,features_test,label_train,label_test = train_test_split(features,label,test_size=0.25)
    version = 0
    while version < 8:
        if version == 0:
            dt_clf = DecisionTreeClassifier(criterion='gini')
            dt_clf.fit(features_train,label_train)
            label_pred = dt_clf.predict(features_test)
            score = accuracy_score(label_test,label_pred)
            crit_gini += score
        if version == 1:
            dt_clf = DecisionTreeClassifier(criterion='entropy')
            dt_clf.fit(features_train,label_train)
            label_pred = dt_clf.predict(features_test)
            score = accuracy_score(label_test,label_pred)
            crit_entropy += score
        if version == 2:
            dt_clf = DecisionTreeClassifier(max_depth=10)
            dt_clf.fit(features_train,label_train)
            label_pred = dt_clf.predict(features_test)
            score = accuracy_score(label_test,label_pred)
            max_dep_10 += score
        if version == 3:
            dt_clf = DecisionTreeClassifier(max_depth=3)
            dt_clf.fit(features_train,label_train)
            label_pred = dt_clf.predict(features_test)
            score = accuracy_score(label_test,label_pred)
            max_dep_3 += score
            if run == 49:
                dot_data = tree.export_graphviz(dt_clf, 'tree.dot', feature_names = feature_name)
        if version == 4:
            dt_clf = DecisionTreeClassifier(min_samples_split=3)
            dt_clf.fit(features_train,label_train)
            label_pred = dt_clf.predict(features_test)
            score = accuracy_score(label_test,label_pred)
            min_split_3 += score
        if version == 5:
            dt_clf = DecisionTreeClassifier(min_samples_split=10)
            dt_clf.fit(features_train,label_train)
            label_pred = dt_clf.predict(features_test)
            score = accuracy_score(label_test,label_pred)
            min_split_10 += score
        if version == 6:
            dt_clf = DecisionTreeClassifier(max_features='auto')
            dt_clf.fit(features_train,label_train)
            label_pred = dt_clf.predict(features_test)
            score = accuracy_score(label_test,label_pred)
            max_f_auto += score
        if version == 7:
            dt_clf = DecisionTreeClassifier(max_features='sqrt')
            dt_clf.fit(features_train,label_train)
            label_pred = dt_clf.predict(features_test)
            score = accuracy_score(label_test,label_pred)
            max_f_sqrt += score
        if run < 5:
            print(version,': ',score)
        version+=1
    run += 1

print('criterion="gini": ',round(crit_gini/50,3))
print('criterion="entropy": ',round(crit_entropy/50,3))
print('max_depth=10: ',round(max_dep_10/50,3))
print('max_depth=3: ',round(max_dep_3/50,3))
print('min_samples_split=3: ',round(min_split_3/50,3))
print('min_samples_split=10: ',round(min_split_10/50,3))
print('max_features="auto": ',round(max_f_auto/50,3))
print('max_features="sqrt": ',round(max_f_sqrt/50,3))

