'''
This part of the code tests the accuracy of Multinomial Naive Bayes algorithm over different versions of the Count vectorizer
classified data
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

legal_text = pd.read_csv('legal_text.csv')
legal_text.dropna(inplace=True)
text_features = legal_text['case_text']
text_labels = legal_text['case_outcome']
feature_train,feature_test,label_train,label_test = train_test_split(text_features,text_labels,test_size=0.3)
mnb_model = MultinomialNB()
version = 0

while version<8:
    print("--------------------VERSION ",(version+1),"--------------------")
    if(version == 0):
        c_vector = CountVectorizer(ngram_range=(1, 1),max_features=2)
    if(version == 1):
        c_vector = CountVectorizer(ngram_range=(2, 2),max_features=2)
    if(version == 2):
        c_vector = CountVectorizer(ngram_range=(1, 1),max_features=6)
    if(version == 3):
        c_vector = CountVectorizer(ngram_range=(2, 2),max_features=6)
    if(version == 4):
        c_vector = CountVectorizer(stop_words='english',ngram_range=(2, 2),max_features=6)
    if(version == 5):
        c_vector = CountVectorizer(stop_words='english',ngram_range=(1, 1),max_features=6)
    if(version == 6):
        c_vector = CountVectorizer(stop_words='english',ngram_range=(2, 2),max_features=2)
    if(version == 7):
        c_vector = CountVectorizer(stop_words='english',ngram_range=(1, 1),max_features=2)
    
    v_mnb_model = Pipeline(steps=[('vectorizer', c_vector), ('classifier', mnb_model)])
    v_mnb_model.fit(feature_train, label_train)
    label_pred = v_mnb_model.predict(feature_test)
    cm = confusion_matrix(label_test,label_pred)
    print("Confusion Matrix: \n",cm)
    fp = cm.sum(axis=0) - np.diag(cm)  
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    print("Accuracy Score: ",accuracy_score(label_test,label_pred))
    print("Precision score: ",(precision_score(label_test,label_pred,average=None,zero_division=1)).mean())
    print("Recall score: ",(recall_score(label_test,label_pred,average=None,zero_division=1)).mean())
    print("Micro-Averaged Accuracy: ",((tp+tn)/(tp+tn+fp+fn)).mean())
    precision = np.nan_to_num(x=(tp/(tp+fp)),nan=0.0)
    recall = np.nan_to_num(x=(tp/(tp+fn)),nan=0.0)
    print("Micro-Averaged Precision Score: ",(precision).mean())
    print("Micro-Averaged Recall Score: ",(recall).mean())
    version += 1
    

