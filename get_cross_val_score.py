# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:13:06 2021

@author: Toprak
"""

seed_value= 43

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import numpy as np

import pickle

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from read_arff import arff_to_numpy

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

def get_ensemble_score(estimators,x,y):
    for i in range(len(estimators)):
        if i == 0:
            a = estimators[i].predict_proba(x)
        else:
            a += estimators[i].predict_proba(x)
        # print(a)
    
    a /= len(estimators)
    
    b = a.argmax(axis=1)
    
    # print(accuracy_score(y, b))
    
    return accuracy_score(y, b)

datasets = ['labor','zoo','lymph','iris','hepatitis','audiology','autos','glass','sonar',
            'heart-statlog','breast-cancer','primary-tumor','ionosphere','colic','vote',
            'balance-scale','soybean','credit-a','breast-w','diabetes','vehicle','anneal',
            'vowel','credit-g','col10','segment','splice','kr-vs-kp','hypothyroid','sick',
            'abalone','waveform','d159','ringnorm','mushroom','letter']

# RF_cross_val_scores = []

# for i, data in enumerate(datasets[:]):
#     X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
#     meta_scores = []
#     kf = KFold(n_splits=3, shuffle=True)
#     for train_index, test_index in kf.split(X):
#         scores1 = []
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         RF = RandomForestClassifier(max_depth = None, n_estimators = 100,n_jobs = -1)
#         RF.fit(X_train,y_train)
#         estimators = RF.estimators_     
#         for k in range(10,101,10):
#             a = get_ensemble_score(estimators[:k],X_test,y_test)
#             scores1.append(a)
#         meta_scores.append(scores1)
#     meta_scores = np.array(meta_scores)
#     s = np.mean(meta_scores,axis = 0).tolist()
#     RF_cross_val_scores.append(s)
# pickle.dump(RF_cross_val_scores, open('rf_fullagac_cross_val_scores', 'wb'))
        
# bag_cross_val_scores = []

# for i, data in enumerate(datasets[:]):
#     X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
#     meta_scores = []
#     kf = KFold(n_splits=3, shuffle=True)
#     for train_index, test_index in kf.split(X):
#         scores1 = []
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         RF = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = None), n_estimators = 100,n_jobs = -1)
#         RF.fit(X_train,y_train)
#         estimators = RF.estimators_     
#         for k in range(10,101,10):
#             a = get_ensemble_score(estimators[:k],X_test,y_test)
#             scores1.append(a)
#         meta_scores.append(scores1)
#     meta_scores = np.array(meta_scores)
#     s = np.mean(meta_scores,axis = 0).tolist()
#     bag_cross_val_scores.append(s)
# pickle.dump(bag_cross_val_scores, open('bag_fullagac_cross_val_scores', 'wb'))
        
# et_cross_val_scores = []

# for i, data in enumerate(datasets[:]):
#     X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
#     meta_scores = []
#     kf = KFold(n_splits=3, shuffle=True)
#     for train_index, test_index in kf.split(X):
#         scores1 = []
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         RF = ExtraTreesClassifier(max_depth = None, n_estimators = 100,n_jobs = -1)
#         RF.fit(X_train,y_train)
#         estimators = RF.estimators_     
#         for k in range(10,101,10):
#             a = get_ensemble_score(estimators[:k],X_test,y_test)
#             scores1.append(a)
#         meta_scores.append(scores1)
#     meta_scores = np.array(meta_scores)
#     s = np.mean(meta_scores,axis = 0).tolist()
#     et_cross_val_scores.append(s)
# pickle.dump(et_cross_val_scores, open('et_fullagac_cross_val_scores', 'wb'))


RF_cross_val_scores = []

for i, data in enumerate(datasets[:]):
    X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
    meta_scores = []
    kf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in kf.split(X):
        scores1 = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for k in [1,2,3,4,5,6,7,8,9,10,None]:
            RF = RandomForestClassifier(max_depth = k, n_estimators = 50,n_jobs = -1)
            RF.fit(X_train,y_train)
            estimators = RF.estimators_
            a = get_ensemble_score(estimators[:k],X_test,y_test)
            scores1.append(a)
        meta_scores.append(scores1)
    meta_scores = np.array(meta_scores)
    s = np.mean(meta_scores,axis = 0).tolist()
    RF_cross_val_scores.append(s)
pickle.dump(RF_cross_val_scores, open('rf_sabit2_cross_val_scores', 'wb'))

RF_cross_val_scores = []

for i, data in enumerate(datasets[:]):
    X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
    meta_scores = []
    kf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in kf.split(X):
        scores1 = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for k in [1,2,3,4,5,6,7,8,9,10,None]:
            RF = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = k), n_estimators = 50,n_jobs = -1)
            RF.fit(X_train,y_train)
            estimators = RF.estimators_
            a = get_ensemble_score(estimators[:k],X_test,y_test)
            scores1.append(a)
        meta_scores.append(scores1)
    meta_scores = np.array(meta_scores)
    s = np.mean(meta_scores,axis = 0).tolist()
    RF_cross_val_scores.append(s)
pickle.dump(RF_cross_val_scores, open('bag_sabit2_cross_val_scores', 'wb'))

RF_cross_val_scores = []

for i, data in enumerate(datasets[:]):
    X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
    meta_scores = []
    kf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in kf.split(X):
        scores1 = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for k in [1,2,3,4,5,6,7,8,9,10,None]:
            RF = ExtraTreesClassifier(max_depth = k, n_estimators = 50,n_jobs = -1)
            RF.fit(X_train,y_train)
            estimators = RF.estimators_
            a = get_ensemble_score(estimators[:k],X_test,y_test)
            scores1.append(a)
        meta_scores.append(scores1)
    meta_scores = np.array(meta_scores)
    s = np.mean(meta_scores,axis = 0).tolist()
    RF_cross_val_scores.append(s)
pickle.dump(RF_cross_val_scores, open('et_sabit2_cross_val_scores', 'wb'))











