'''
This part of the code tests the accuracy of Gaussian Naive Bayes algorithm over different split versions of the data
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

frogs = pd.read_csv('frogs.csv')
features = frogs.drop(columns='Green frogs',axis=1)
labels = frogs['Green frogs']
run = 0
train_score_total = 0
test_score_total = 0
label_pred_prob_total = 0
pred_total = 0
label_total = 0
pred_prob_total = 0
while run<50:
    feature_train,feature_test,label_train,label_test = train_test_split(features,labels,test_size=0.3)
    gnb_model = GaussianNB()
    gnb_model = gnb_model.fit(feature_train,label_train)
    train_score = gnb_model.score(feature_train,label_train)
    train_score_total += train_score
    test_score = gnb_model.score(feature_test,label_test)
    test_score_total += test_score
    label_pred_prob = gnb_model.predict_proba(feature_test).max(axis=1)
    label_pred_prob_total += label_pred_prob.mean()
    low_confidence = label_pred_prob<(label_pred_prob.mean())
    label_pred = gnb_model.predict(feature_test)
    pred = (label_pred[low_confidence]).mean()
    pred_total += pred
    label = (label_test[low_confidence]).mean()
    label_total += label
    pred_prob = (label_pred_prob[low_confidence]).mean()
    pred_prob_total += pred_prob
    run += 1
print("Training Data Score: ",train_score_total/50)
print("Testing Data Score: ",test_score_total/50)
print("Test Data Probabilities for Predictions: ",label_pred_prob_total/50)
print("Low Confidence Predictions: ",pred_total/50)
print("Correct Predictions: ",label_total/50)
print("Associated Probabilities: ",pred_prob_total/50)

