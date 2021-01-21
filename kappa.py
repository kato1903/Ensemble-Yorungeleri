seed_value= 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from read_arff import arff_to_numpy
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

import numpy as np

import random

import matplotlib.cm as cm

import pickle

datasets = ['labor','zoo','lymph','iris','hepatitis','audiology','autos','glass','sonar',
            'heart-statlog','breast-cancer','primary-tumor','ionosphere','colic','vote',
            'balance-scale','soybean','credit-a','breast-w','diabetes','vehicle','anneal',
            'vowel','credit-g','col10','segment','splice','kr-vs-kp','hypothyroid','sick',
            'abalone','waveform','d159','ringnorm','mushroom','letter']

def isNaN(num):
    return num != num

def get_kappas(estimators,X,y):
    points = []
    for i in range(len(estimators)):
        for j in range(i):
            pred1 = estimators[i].predict(X)
            pred2 = estimators[j].predict(X)            
            kappa = cohen_kappa_score(pred1,pred2)
            score1 = 1 - accuracy_score(y,pred1)
            score2 = 1 - accuracy_score(y,pred2)
            ort_score = (score1 + score2) / 2
            # print(pred1)
            # print(pred2)
            if isNaN(kappa):
                kappa = 1.0
            points.append([kappa,ort_score])
    return points

from sklearn.metrics import accuracy_score

def get_ensemble_score(estimators,x,y):
    print('girdi')
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

def get_all_ensemble_score(estimators,x,y):
    scores = []
    for i in range(10,len(estimators)+1,10):
        # index = int((i) * 10)
        # print(len(estimators))
        # print('Len',len(estimators[:i]))
        score = get_ensemble_score(estimators[:i],x,y)
        # print('estimators',estimators[:index])
        scores.append(score)
    return scores
    







import matplotlib.pyplot as plt
def plot(meta_points):

    means = []
    
    for i in meta_points:
        points = np.array(i)
        means.append([np.mean(points[:,1]),np.mean(points[:,0])])
    
    means = np.array(means)
    
    
    # fig, ax = plt.subplots()
    plt.xlabel('Error')
    plt.ylabel('Kappa')
    plt.margins(0.5)
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.plot(means[:,0],means[:,1])
    plt.scatter(means[:,0],means[:,1])
    
    for i, txt in enumerate(range(10,100,10)):
        plt.annotate(txt, (means[i,0], means[i,1]),xytext=(10,10), textcoords='offset points' )
    
    plt.show()
    
    plt.close()
    plt.clf()

def plot_one_point(alg,meta_points,which_points,dataset,isAnnotate,isLegend,scores,scores2):
    import matplotlib.pyplot as plt
    
    plt.figure(num=None, figsize=(8, 6), dpi=160, facecolor='w', edgecolor='k')
    # plt.margins(0.2)
    # plt.ylim(0.5,0.9)
    # plt.xlim(0,0.5)
    # plt.ylim(ymax=0.95)
    # plt.ylim(0.4,0.95)
    title = str(alg) + ' ' + str(dataset)
    plt.title(title)
    
    colors = cm.rainbow(np.linspace(0, 1, len(which_points)))
    
    listex = []
    listey = []
    
    k = 0
    
    for i,n,c in zip(meta_points,which_points,colors):
        points = np.array(i)
        plt.xlabel('Error')
        plt.ylabel('Kappa')
        # plt.xlim(0,1)
        # plt.ylim(0,1)
        # print(n)
        
        
        if (n+1) % 10 == 0:
            plt.scatter(np.mean(points[:,1]),np.mean(points[:,0]),
                        label = str(n+1) + ' - ' + str(format(scores2[(n+1)//10-1], '.2f')) + ' - ' + str(format(scores[(n+1)//10-1], '.2f')), 
                        color=c,alpha=0.5)
        else:
            plt.scatter(np.mean(points[:,1]),np.mean(points[:,0]),label = None,color=c,alpha=0.5)
        listex.append(np.mean(points[:,1]))
        listey.append(np.mean(points[:,0]))
        if isLegend:
            plt.legend()
        if isAnnotate:
            plt.annotate(n, (np.mean(points[:,1]), np.mean(points[:,0])),xytext=(5,5), textcoords='offset points' )
        
        k += 1
    # plt.plot(listex,listey)
    plt.legend()
    r_title = title.replace('\n','')
    plt.savefig(str(r_title) + '_single_points.png')
    plt.show()

def plot_one_point2(alg,meta_points,which_points,dataset,isAnnotate,isLegend,scores,scores2):
    import matplotlib.pyplot as plt
    
    plt.figure(num=None, figsize=(8, 6), dpi=160, facecolor='w', edgecolor='k')
    # plt.margins(0.2)
    # plt.ylim(0.5,0.9)
    # plt.xlim(0,0.5)
    # plt.ylim(ymax=0.95)
    # plt.ylim(0.4,0.95)
    title = str(alg) + ' ' + str(dataset)
    plt.title(title)
    
    colors = cm.rainbow(np.linspace(0, 1, len(which_points)))
    
    listex = []
    listey = []
    
    k = 0
    
    for i,n,c in zip(meta_points,which_points,colors):
        points = np.array(i)
        plt.xlabel('Error')
        plt.ylabel('Kappa')
        # plt.xlim(0,1)
        # plt.ylim(0,1)
        
        print(n)
        
        if n != None:
            plt.scatter(np.mean(points[:,1]),np.mean(points[:,0]),
                        label = str(n) + ' - ' + str(format(scores2[n-1], '.2f')) + ' - ' + str(format(scores[n-1], '.2f'))
                        , color=c,alpha=0.5)
            print(np.mean(points[:,1]),np.mean(points[:,0]))
        else:
            plt.scatter(np.mean(points[:,1]),np.mean(points[:,0]),
                        label = 'Sonsuz - ' + ' - ' + str(format(scores2[10], '.2f')) + ' - ' + str(format(scores[10], '.2f'))
                        ,color=c,alpha=0.5)
            
        listex.append(np.mean(points[:,1]))
        listey.append(np.mean(points[:,0]))
        if isLegend:
            plt.legend()
        if isAnnotate:
            plt.annotate(str(n), (np.mean(points[:,1]), np.mean(points[:,0])),xytext=(5,5), textcoords='offset points' )
        
        k += 1
    # plt.plot(listex,listey)
    plt.legend()
    r_title = title.replace('\n','')
    plt.savefig(str(r_title) + '_single_points.png')
    plt.show()



def plot_multiple_point(alg,meta_points,which_points,dataset):
    import matplotlib.pyplot as plt
    # plt.figure(num=None, figsize=(8, 6), dpi=160, facecolor='w', edgecolor='k')
    plt.figure(dpi=160)
    # plt.margins(0.2)
    # plt.ylim(ymax=0.95)
    # plt.ylim(0.4,0.95)
    # plt.xlim(0,0.5)
    title = str(alg) + ' ' + str(dataset)
    plt.title(title)
    
    colors = cm.rainbow(np.linspace(0, 1, len(which_points)))
    
    for i,n,c in zip(meta_points,which_points,colors):
    
        # print(n)
        points = np.array(i)
        plt.xlabel('Error')
        plt.ylabel('Kappa')
        plt.scatter(points[:,1],points[:,0],label = str(n), alpha = 0.01, color = c)

    leg = plt.legend()
    
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    
    plt.savefig(title.replace('\n','') + '_multiple_points.png')
    plt.show()

def plot_different_alg_multiple_point(meta_meta_points,which_points):
    import matplotlib.pyplot as plt
    plt.figure(num=None, figsize=(8, 6), dpi=160, facecolor='w', edgecolor='k')
    # plt.margins(0.2)
    # plt.xlim(xmax=0.9)
    # plt.ylim(0,0.5)
    plt.ylim(0.4,0.95)
    for name,meta_points in zip(['RandomForest','Bagging','AdaBoostClassifier'],meta_meta_points):
        for i,n in zip(meta_points,which_points):
        
            # print(n)
            points = np.array(i)
            plt.xlabel('Kappa')
            plt.ylabel('Error')
            if name == 'RandomForest' or name == 'Bagging':
                plt.scatter(points[:,0],points[:,1],label = str(n) + '_' + name, alpha = 0.2)
            else:
                plt.scatter(points[:,0],points[:,1],label = str(n) + '_' + name, alpha = 0.2)
    plt.legend()
    plt.show()


        

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

isFullAgac = True
isMaxDepht = True

if isFullAgac:

    which_points = range(1,101,1)
    
    meta_meta_points = []
    
    isRandomForest = True
    isBagging = False
    isAdaBoost = False
    
    isAnnotate = False
    isLegend = False
    
    max_depth = None
    
    for i, data in enumerate(datasets[:]):
        meta_points = []
        meta_points_2 = []
        meta_points2 = []
        meta_points2_2 = []
        meta_points3 = []
        X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
        X, X_test, y, y_test = train_test_split(X,y,stratify = y,test_size=0.33)
        
        if isRandomForest:
            RF = ExtraTreesClassifier(max_depth = max_depth, n_estimators = which_points[-1],n_jobs = -1)
            RF.fit(X,y)
            estimators = RF.estimators_        
            # print(len(estimators))
            points = get_kappas(estimators,X,y)
            points_2 = get_kappas(estimators,X_test,y_test)
            for ii in which_points[:-1]:
                index = int((ii * (ii + 1)) / 2)
                meta_points.append(points[:index])
                meta_points_2.append(points_2[:index])
            meta_meta_points.append(meta_points)
            scores = get_all_ensemble_score(estimators,X,y)
            scores2 = get_all_ensemble_score(estimators,X_test,y_test)
            # plot_multiple_point('RandomForest',meta_points,which_points,data)
            plot_one_point('ExtraTrees Full Ağaç Tekil Öğrenici Sayısına Göre Kappa - Error \n Training Dataset ',
                           meta_points,which_points,data,isAnnotate,isLegend,scores,scores2)
            plot_one_point('ExtraTrees Full Ağaç Tekil Öğrenici Sayısına Göre Kappa - Error \n Test Dataset ',
               meta_points_2,which_points,data,isAnnotate,isLegend,scores,scores2)

            pickle.dump(meta_points, open(str(data)+'fullagac_et_meta_points.pkl', 'wb'))
            pickle.dump(meta_points_2, open(str(data)+'fullagac_et_meta_points2.pkl', 'wb'))
            pickle.dump(scores, open(str(data)+'fullagac_et_score.pkl', 'wb'))
            pickle.dump(scores2, open(str(data)+'fullagac_et_score2.pkl', 'wb'))
 
        if isBagging:
            Bagg = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = max_depth),n_estimators = which_points[-1],n_jobs = 5)
            Bagg.fit(X,y)
            estimators = Bagg.estimators_        
            # print(len(estimators))
            points = get_kappas(estimators,X,y)
            points_2 = get_kappas(estimators,X_test,y_test)
            for ii in which_points[:-1]:
                index = int((ii * (ii + 1)) / 2)
                meta_points2.append(points[:index])
                meta_points2_2.append(points_2[:index])
            meta_meta_points.append(meta_points2)
            # plot_multiple_point('RandomForest',meta_points,which_points,data)
            scores = get_all_ensemble_score(estimators,X,y)
            scores2 = get_all_ensemble_score(estimators,X_test,y_test)
            pickle.dump(meta_points2, open(str(data)+'fullagac_bag_meta_points.pkl', 'wb'))
            pickle.dump(meta_points2_2, open(str(data)+'fullagac_bag_meta_points2.pkl', 'wb'))
            pickle.dump(scores, open(str(data)+'fullagac_bag_score.pkl', 'wb'))
            pickle.dump(scores2, open(str(data)+'fullagac_bag_score2.pkl', 'wb'))
            
            plot_one_point('Bagging Full Ağaç Tekil Öğrenici Sayısına Göre Kappa - Error \n Training Dataset ',meta_points2,which_points,data,isAnnotate,isLegend
                           ,scores,scores2)

            plot_one_point('Bagging Full Ağaç Tekil Öğrenici Sayısına Göre Kappa - Error \n Test Dataset ',meta_points2_2,which_points,data,isAnnotate,isLegend
                           ,scores,scores2)
            
        if isAdaBoost:
            AdaBoost = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = max_depth), n_estimators = which_points[-1])
            AdaBoost.fit(X,y)
            estimators = AdaBoost.estimators_        
            # print(len(estimators))
            points = get_kappas(estimators,X,y)
            for ii in which_points[:-1]:
                index = int((ii * (ii + 1)) / 2)
                meta_points3.append(points[:index])
            meta_meta_points.append(meta_points3)
            # plot_multiple_point('RandomForest',meta_points,which_points,data)
            plot_one_point('AdaBoost',meta_points3,which_points,data,isAnnotate,isLegend)        
        # 1 / 0

1 / 0

if isMaxDepht:

    meta_meta_points = []
    isRandomForest = True
    isBagging = False
    isAdaBoost = False
    max_depths = [1,2,3,4,5,6,7,8,9,10,None]
    
    n_estimator = 50
    
    isAnnotate = True
    isLegend = False
    
    for i, data in enumerate(datasets[:]):
        meta_points = []
        meta_points_2 = []
        meta_points2 = []
        meta_points2_2 = []
        meta_points3 = []
        X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
        X, X_test, y, y_test = train_test_split(X,y,stratify = y,test_size=0.33)
        score = []
        score2 = []
        if isRandomForest:
            for max_depth in max_depths:
                RF = ExtraTreesClassifier(max_depth = max_depth, n_estimators = n_estimator,n_jobs=-1)
                RF.fit(X,y)
                estimators = RF.estimators_        
                # print(len(estimators))
                points = get_kappas(estimators,X,y)
                points_2 = get_kappas(estimators,X_test,y_test)
                meta_points.append(points[:])
                meta_points_2.append(points_2[:])
                y_pred1 = RF.predict(X)
                y_pred2 = RF.predict(X_test)
                s1 = accuracy_score(y, y_pred1)
                s2 = accuracy_score(y_test, y_pred2)
                score.append(s1)
                score2.append(s2)                
                
            meta_meta_points.append(meta_points)
            plot_one_point2('ExtraTrees Sabit Tekil Öğrenici Maximum Derinliğe Göre Kappa - Error \n Train Dataset ',
                           meta_points,max_depths,data,isAnnotate,isLegend,score,score2)
            plot_one_point2('ExtraTrees Sabit Tekil Öğrenici Maximum Derinliğe Göre Kappa - Error \n Test Dataset ',
               meta_points_2,max_depths,data,isAnnotate,isLegend,score,score2)
            with open(str(data)+'sabitensemble_et_meta_points.pkl', 'wb') as f:
                pickle.dump(meta_points, f)
            with open(str(data)+'sabitensemble_et_meta_points2.pkl', 'wb') as f:
                pickle.dump(meta_points_2, f)    
            with open(str(data)+'sabitensemble_et_score.pkl', 'wb') as f:
                pickle.dump(score, f)
            with open(str(data)+'sabitensemble_et_score2.pkl', 'wb') as f:
                pickle.dump(score2, f)
        score = []
        score2 = []
        if isBagging:
            for max_depth in max_depths:
                Bagg = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = max_depth),n_estimators = n_estimator,n_jobs = 5)
                Bagg.fit(X,y)
                estimators = Bagg.estimators_        
                # print(len(estimators))
                points = get_kappas(estimators,X,y)
                points_2 = get_kappas(estimators,X_test,y_test)
                meta_points2.append(points[:])
                meta_points2_2.append(points_2[:])
                y_pred1 = Bagg.predict(X)
                y_pred2 = Bagg.predict(X_test)
                s1 = accuracy_score(y, y_pred1)
                s2 = accuracy_score(y_test, y_pred2)
                score.append(s1)
                score2.append(s2)       

            meta_meta_points.append(meta_points2)
            plot_one_point2('Bagging Sabit Tekil Öğrenici Maximum Derinliğe Göre Kappa - Error \n Train Dataset ',
                           meta_points2,max_depths,data,isAnnotate,isLegend,score,score2)
            plot_one_point2('Bagging Sabit Tekil Öğrenici Maximum Derinliğe Göre Kappa - Error \n Test Dataset ',
                           meta_points2_2,max_depths,data,isAnnotate,isLegend,score,score2)            
            with open(str(data)+'sabitensemble_bag_meta_points.pkl', 'wb') as f:
                pickle.dump(meta_points, f)
            with open(str(data)+'sabitensemble_bag_meta_points2.pkl', 'wb') as f:
                pickle.dump(meta_points_2, f)    
            with open(str(data)+'sabitensemble_bag_score.pkl', 'wb') as f:
                pickle.dump(score, f)
            with open(str(data)+'sabitensemble_bag_score2.pkl', 'wb') as f:
                pickle.dump(score2, f) 
            
    
        if isAdaBoost:
            for max_depth in max_depths[:]:
                try:
                    AdaBoost = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = max_depth), n_estimators = n_estimator)
                    AdaBoost.fit(X,y)
                    estimators = AdaBoost.estimators_        
                    # print(len(estimators))
                    points = get_kappas(estimators,X,y)
                    meta_points3.append(points[:])
                except:
                    pass
            meta_meta_points.append(meta_points3)
            plot_one_point2('AdaBoost Sabit Tekil Öğrenici Maximum Derinliğe Göre Kappa - Error \n Dataset: ',
                           meta_points3,max_depths,data,isAnnotate,isLegend)


1 / 0






for i, data in enumerate(datasets[:10]):
    meta_points = []
    meta_points2 = []
    meta_points3 = []
    for ii,n in enumerate(which_points):
        X, y = arff_to_numpy('Datasets/'+str(data)+'.arff')
        
        if isRandomForest:
            if ii == 0:
                RF = RandomForestClassifier(max_depth = 10,n_estimators = which_points[-1],n_jobs = 5)
                RF.fit(X,y)
                estimators = RF.estimators_
            print(len(estimators[:]))
            points = get_kappas(estimators[:],X,y)
            
            for j in range(which_points[-1]):
                meta_points.append(points)
            1 / 0
        if isBagging:
            if ii == 0:
                BC = BaggingClassifier(base_estimator = DecisionTreeClassifier(max_depth = 10),n_estimators = which_points[-1],n_jobs = 5)
                BC.fit(X,y)
                estimators = BC.estimators_
            print(len(estimators[:n]))
            points = get_kappas(estimators[:n],X,y)
            meta_points2.append(points)
        
        if isAdaBoost:
            if ii == 0:
                ABC = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 10), n_estimators = which_points[-1])
                ABC.fit(X,y)
                estimators = ABC.estimators_
            print(len(estimators[:n]))
            points = get_kappas(estimators[:n],X,y)
            meta_points3.append(points)
    
    if isRandomForest:
        meta_meta_points.append(meta_points)
    if isBagging: 
        meta_meta_points.append(meta_points2)
    if isAdaBoost: 
        meta_meta_points.append(meta_points3)
    


# plot_different_alg_multiple_point(meta_meta_points,which_points)
    if isRandomForest:
        # plot_multiple_point('RandomForest',meta_points,which_points,data)
        plot_one_point('RandomForest',meta_points,which_points,data,isAnnotate,isLegend)
    if isBagging:
        # plot_multiple_point('Bagging',meta_points2,which_points,data)
        plot_one_point('Bagging',meta_points2,which_points,data,isAnnotate,isLegend)
    if isAdaBoost:
        # plot_multiple_point('AdaBoost',meta_points3,which_points,data)
        plot_one_point('AdaBoost',meta_points3,which_points,data,isAnnotate,isLegend)

    # plot(meta_points)
    # plot_one_point(meta_points,which_points)

# import matplotlib.pyplot as plt
# plt.figure(num=None, figsize=(8, 6), dpi=160, facecolor='w', edgecolor='k')
# plt.margins(0.2)
# for i,n in zip(meta_points,which_points):

#     print(n)
#     points = np.array(i)
#     plt.xlabel('Error')
#     plt.ylabel('Kappa')
#     plt.scatter(points[:,1],points[:,0],label = str(n), alpha = 0.1)
# plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# plt.figure(num=None, figsize=(8, 6), dpi=160, facecolor='w', edgecolor='k')
# for i,n in zip(meta_points,which_points):
#     points = np.array(i)
#     plt.xlabel('Error')
#     plt.ylabel('Kappa')
#     # plt.xlim(0,1)
#     # plt.ylim(0,1)
#     plt.scatter(np.mean(points[:,1]),np.mean(points[:,0]),label = n)
#     plt.legend()
#     plt.annotate(n, (np.mean(points[:,1]), np.mean(points[:,0])),xytext=(5,5), textcoords='offset points' )

# plt.show()




















