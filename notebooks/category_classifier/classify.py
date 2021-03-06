# -*- coding: utf-8 -*-
"""classify_case.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u3qil-3-1U5ahglE1EVCZQMVi4jMyV-9
"""

import pandas as pd
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import svm

#Read dataset and remove NA values
report_1 = pd.read_csv('datasets/safecity_reports_07082019.csv')
report_1 = report_1[pd.notna(report_1['DESCRIPTION'])]

#Check the number of rows without a location (value==NaN)
print('Shape of the DataFrame used for training : ', report_1.shape)

categories=[]
for index, row in report_1.iterrows():
    cat_group_list = [i for i in row.CATEGORY.split(',')]
    del(cat_group_list[-1])
    for category in cat_group_list:
      category = [category.lstrip()]
      if category not in categories:
        categories.append(category) #add a new category
      else:
        next
    #replace in the original df with the list. iat[] is to set a single row and 5 is the CATEGORY column index    
    if index == report_1.shape[0]:
      break
    report_1.iat[index, 5] = cat_group_list
    
print("Total number of unique categories : {}\n".format(len(categories)))
counts = [0] * len(categories)
category_list = [cat[0] for cat in categories]

#Select x and Y (without HumanTrafficking or PettyRobbery is til column 28)
x_report_1 = report_1[report_1.columns[4]]
y_report_1 = report_1[report_1.columns[14:28]]
X_train, X_test, y_train, y_test = train_test_split(x_report_1, y_report_1, test_size=0.2, random_state=17)
categories = y_train.columns

def execute(pipeline, X_train=X_train, X_test=X_test, confusion_matrix=False, verbose=True):

  #This function executes a pipeline or a single test string"

  accuracies=[]
  print("Generating predictions...")
  for category in categories:
    pipeline.fit(X_train, y_train[category])
    prediction = pipeline.predict(X_test)
    
    if len(X_test) == 1:
      print('Prediction for {} is {}'.format(category, prediction)) if verbose else None
    else:
      print('Test accuracy for {} is {}'.format(category, metrics.accuracy_score(y_test[category], prediction))) if verbose else None
      accuracies.append(metrics.accuracy_score(y_test[category], prediction))
      print(metrics.confusion_matrix(y_test[category], prediction, labels=[0,1])) if confusion_matrix else None
      print('precision_recall_fscore_support_weighted', precision_recall_fscore_support(y_test[category], prediction, average='weighted')) if verbose else None

  print('mean: ', sum(accuracies)/len(accuracies)) if verbose and len(accuracies)!=0 else None
  return accuracies

SVM_pipeline = Pipeline([('vectorizer', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', OneVsRestClassifier(svm.SVC(kernel='linear', gamma='auto', C=1.0)))])

test_string = [input("Describe the case: ")]
print('...................................................................\n')
#test_string = [""""He started rubbing my shoulders, telling me I looked stressed. Then he went down in my shirt. 
#He walked around the living room area and he came back. That is when he touched my breast. 
#Then he grabbed my waist of my pants and also grabbed my hair as I tried to leave. I made it to the door and left. 
#It happened so quickly. I was just trying to get out of the townhouse"""]

acc = execute(SVM_pipeline, X_test = test_string)