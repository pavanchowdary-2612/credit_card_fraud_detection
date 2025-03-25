# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 22:27:07 2025

@author: pavan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve


data = pd.read_csv(r"C:\Users\pavan\Downloads\creditcard(1).csv")

X = data.drop(columns=["Class"])
y = data["Class"]

data = data.dropna(subset=['Class'])
X = data.drop(columns=["Class"])
y = data["Class"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#----------------Linear Regression-------------------

lr = LinearRegression()
lr.fit(X_train_scaled,y_train)
y_pred = (lr.predict(X_test_scaled)>=0.5).astype(int)

print("\n--- Linear Regression-----")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:",roc_auc_score(y_test,lr.predict(X_test_scaled)))


#-----------------------Decision Tree-----------------------

param_dt = {'max_depth':[3,5,7,9],'min_samples_split':[2,5,10]}
dt=GridSearchCV(DecisionTreeClassifier(random_state=0), param_dt,cv=3,scoring='roc_auc')
dt.fit(X_train,y_train)
print("\n----Decision Tree----")
print("Best Params:",dt.best_params_)
print(confusion_matrix(y_test, dt.predict(X_test)))
print(classification_report(y_test, dt.predict(X_test)))
print("ROC-AUC:",roc_auc_score(y_test,dt.predict(X_test)))

#--------------- SVM ---------------

param_svm = {'C':[0.1,1,10],'kernel':['linear','rbf']}
svm = GridSearchCV(SVC(probability=True, random_state=0), param_svm, cv=3, scoring='roc_auc')
svm.fit(X_train_scaled,y_train)
print("\n--- SVM ---")
print("Best Params:", svm.best_params_)
print(confusion_matrix(y_test, svm.predict(X_test_scaled)))
print(classification_report(y_test, svm.predict(X_test_scaled)))
print("ROC-AUC:", roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1]))

#-------------Gradient Boosting---------------

param_gb = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
gb = GridSearchCV(GradientBoostingClassifier(random_state=0), param_gb, cv=3, scoring='roc_auc')
gb.fit(X_train, y_train)
print("\n--- Gradient Boosting ---")
print("Best Params:", gb.best_params_)
print(confusion_matrix(y_test, gb.predict(X_test)))
print(classification_report(y_test, gb.predict(X_test)))
print("ROC-AUC:", roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1]))








plt.figure(figsize=(10,6))

def plot_roc(model,X,label):
    y_proba = model.predict_proba(X)[:,1]
    fpr,tpr,_ = roc_curve(y_test,y_proba)
    plt.plot(fpr,tpr,label=label)


plot_roc(dt.best_estimator_,X_test,"Decision Tree")
plot_roc(svm.best_estimator_,X_test_scaled,"SVM")
plot_roc(gb.best_estimator_,X_test,"Gradient Boosting")
plt.plot([0,1],[0,1],linestyle='--',color="green")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positve Rate")
plt.legend()
plt.title("ROC Curves for Models")
plt.show()




















