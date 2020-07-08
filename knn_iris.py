# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:19:49 2020

@author: PRIYANSH knn
"""

import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score , precision_score , roc_auc_score ,roc_curve

a=pd.read_csv(open('iris.csv','rb'))
x=a.iloc[:,1:5]
y=a.iloc[:,5].values


#here .values is necessary in y to get the output on console for labelencoder otherwise it will show unhashable error

y= LabelEncoder().fit_transform(y)

#the array is reshaped because onehotencoder takes 2D array and we were having 1 D array
y=y.reshape(-1,1)

#some of the variables are coded as 0 1 and 2 but that variables are not define with ranking.
#so dummy variable are introduce inplace of that
#oneHotEncoder for dummy variable creation using scikitlearn

ohe = OneHotEncoder()


y=ohe.fit_transform(y).toarray()


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


#to find the best value of K we have three algos
#1 brute force for smaller no of samples
#2 kd_tree for the  features less than 20
#3 ball_tree for the features more than 20 and higher no of samples
"""
cv_scores = []
neighbors = list(np.arange(3,50,2))
for n in neighbors:
    KNN = knn(n_neighbors = n,algorithm = 'kd_tree')
    
    cross_val = cross_val_score(KNN,xtrain,ytrain,cv = 5 , scoring = 'accuracy')
    cv_scores.append(cross_val.mean())
    
error = [1-x for x in cv_scores]
optimal_n = neighbors[ error.index(min(error)) ]
knn_optimal = knn(n_neighbors = optimal_n,algorithm = 'kd_tree')
knn_optimal.fit(xtrain,ytrain)
pred = knn_optimal.predict(xtest)
acc = accuracy_score(ytest,pred)*100
print("The accuracy for optimal k = {0} using brute is {1}".format(optimal_n,acc))



print(classification_report(ytest,pred))

"""


Knn=knn(n_neighbors=3)

Knn.fit(xtrain, ytrain) 

print(Knn.score(xtest,ytest))
