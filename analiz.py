import pickle

import numpy as np

import pandas as pd

datasets = ['labor','zoo','lymph','iris','hepatitis','audiology','autos','glass','sonar',
            'heart-statlog','breast-cancer','primary-tumor','ionosphere','colic','vote',
            'balance-scale','soybean','credit-a','breast-w','diabetes','vehicle','anneal',
            'vowel','credit-g','col10','segment','splice','kr-vs-kp','hypothyroid','sick',
            'abalone','waveform','d159','ringnorm','mushroom','letter']
ccc = [0,0,0]

alg = 'et'

with open(alg + '_sabit2_cross_val_scores', 'rb') as f:
    a5 = pickle.load(f)

dfs = []

for ii,dataset in enumerate(datasets[:]):

# dataset = 'labor'

    with open(str(dataset)+'sabitensemble_'+alg+'_meta_points.pkl', 'rb') as f:
        a1 = pickle.load(f)
    with open(str(dataset)+'sabitensemble_'+alg+'_meta_points2.pkl', 'rb') as f:
        a2 = pickle.load(f)
    with open(str(dataset)+'sabitensemble_'+alg+'_score.pkl', 'rb') as f:
        a3 = pickle.load(f)
    # with open(str(dataset)+'sabitensemble_'+alg+'_score2.pkl', 'rb') as f:
    #     a4 = pickle.load(f)

    a4 = a5[ii]
    
    # a1 = [np.mean(np.array(i),axis = 0) for ii,i in enumerate(a1) if ii % 10 == 0]
    a1 = [np.mean(np.array(i),axis = 0) for ii,i in enumerate(a1)]
    a2 = [np.mean(np.array(i),axis = 0) for i in a2]
    
    # from sklearn.preprocessing import MinMaxScaler
    
    # scaler = MinMaxScaler()
    
    # a1 = scaler.fit_transform(a1)
    
    # for i in range(10):
    #     print(np.round(np.sqrt(a1[i][0] ** 2 + a1[i][1] ** 2),3), 
    #           np.round(a4[i],3), np.round(a3[i],3), np.round(a4[i],3))
    
    
    
    
    data = {'Kappa': np.array(a1)[:,0],
            'TekilError': np.array(a1)[:,1],
            'TrainEnsembleAcc':a3,
            'TestAcc': a4
            }
    

    
    df = pd.DataFrame(data,columns=['Kappa','TekilError','TrainEnsembleAcc','TestAcc'])
    
    dfs.append(df)
    
    if df.corr().iloc[3,:-1][0] > 0:
        ccc[0] += 1
    if df.corr().iloc[3,:-1][1] > 0:
        ccc[1] += 1
    if df.corr().iloc[3,:-1][2] > 0:
        ccc[2] += 1
    print(df.corr().iloc[3,:-1][0])
    print('----')

deep_dfs = dfs.copy()

k = 0

for i in range(36):
    
    test = dfs.pop(i)
    
    a = pd.concat(dfs)
    
    x = a.iloc[:,:3].values
    y = a.iloc[:,3].values

    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression().fit(x[:], y[:])
    
    # print(reg.score(x,y))
    
    # print(reg.coef_)

    pred = reg.predict(test.iloc[:,:3])

    max_pred = max(pred)
    
    indexler = [i for i, j in enumerate(pred) if j == max_pred]
    
    real_y = test.iloc[:,3].tolist()
    
    max_pred = max(real_y)
    
    indexler2 = [i for i, j in enumerate(real_y) if j == max_pred]    

    dfs = deep_dfs.copy()
    
    # print(indexler)
    
    # print(len(indexler2))    
    
    for ab in indexler:
        if ab in indexler2:
            k += 1
            print(ab)
print(k)


















# with open(str(dataset)+'sabitensemble_rf_meta_points.pkl', 'rb') as f:
#     a5 = pickle.load(f)
# with open(str(dataset)+'sabitensemble_rf_meta_points2.pkl', 'rb') as f:
#     a6 = pickle.load(f)
# with open(str(dataset)+'sabitensemble_rf_score.pkl', 'rb') as f:
#     a7 = pickle.load(f)
# with open(str(dataset)+'sabitensemble_rf_score2.pkl', 'rb') as f:
#     a8 = pickle.load(f)