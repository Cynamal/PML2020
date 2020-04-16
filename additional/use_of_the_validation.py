#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gc
import random
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('run', 'validation.py')
get_ipython().run_line_magic('run', 'features.py')


# In[2]:


try:
    if not df_all.empty:
        print('data already imported')
except:
    train_h5 = pd.read_hdf('../input/train_online_retail.h5')
    test_h5 = pd.read_hdf('../input/test_online_retail.h5')
finally:
    df_all = pd.concat([train_h5, test_h5], sort=False)
    
train, test = prepare_data(df_all)
feats = get_feats(train)


# In[14]:


def draw_importance_features(model, df, plot_type='gain'):
    fscore = model.get_booster().get_score(importance_type=plot_type) #cover, gain, weight
    maps_name = dict([ ("{0}".format(col), col) for i, col in enumerate(df.columns)])
    impdf = pd.DataFrame([ {'feature': maps_name[k], 'importance': fscore[k]} for k in fscore ])
    impdf = impdf.sort_values(by='importance', ascending=False).reset_index(drop=True)
    impdf['importance'] /= impdf['importance'].sum()
    impdf.index = impdf['feature']
    impdf.plot(kind='bar', title=f'{plot_type} - Importance Features', figsize=(15, 14))

scores_mse, scores_f1, scores_recall, scores_precision = [], [], [], []
for id_train, id_test in cross_validation(train, data_perc=1, k_folds=7):
    model = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.3, seed=0)
    X = train.iloc[id_train]
    y = train.iloc[id_train]['is_canceled']
    X_test = train.iloc[id_test]
    y_test = train.iloc[id_test]['is_canceled']
    model.fit(X[feats], y)
    y_pred = model.predict(X_test[feats])
    X_test['total_return'] = X_test['price_total'] * X_test['is_canceled']
    predicted = X_test.copy()
    predicted['is_canceled'] = y_pred
    predicted['total_return'] = predicted['price_total'] * predicted['is_canceled']
    score_f1 = f1_score(X_test['is_canceled'], y_pred)
    score_recall = recall_score(X_test['is_canceled'], y_pred)
    score_precision = precision_score(X_test['is_canceled'], y_pred)
    score_mse = mse(X_test['total_return'], predicted['total_return'])
    scores_mse.append(score_mse)
    scores_f1.append(score_f1)
    scores_recall.append(score_recall)
    scores_precision.append(score_precision)
    print('mse')
    print(score_mse)
    print('f1')
    print(score_f1)
    print('recall')
    print(score_recall)
    print('precision')
    print(score_precision)
    draw_importance_features(model, X[feats], plot_type='gain')
    
print('\n\nsummary')
print('mse')
print(sum(scores_mse) / len(scores_mse))
print('f1')
print(sum(scores_f1) / len(scores_f1))
print('recall')
print(sum(scores_recall) / len(scores_recall))
print('precision')
print(sum(scores_precision) / len(scores_precision))


# duplikaty
# duplikaty, ale inna liczba
# price unit na minusie

# In[11]:


save_results = True
if save_results:
    model = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.2, seed=0)
    model.fit(train[feats], train['is_canceled'])
    test['is_canceled'] = model.predict(test[feats])
    test['total_return'] = test['price_total'] * test['is_canceled']
    result = test.groupby('invoice')['total_return'].agg(np.sum).to_frame().reset_index()
    result[ ['invoice', 'total_return'] ].to_csv('../output/xgb_and_cnt_features_classification.csv', index=False)
    result[ ['invoice', 'total_return'] ]


# In[12]:


regression = False
if regression:
    scores_mse = []
    for id_train, id_test in cross_validation(train, data_perc=1, k_folds=7):
        model = xgb.XGBRegressor(max_depth=5, n_estimators=100, learning_rate=0.2, seed=0, objective='reg:squarederror')
        train['total_return'] = train['price_total'] * train['is_canceled']
        X = train.iloc[id_train]
        y = train.iloc[id_train]['total_return']
        X_test = train.iloc[id_test]
        model.fit(X[feats], y)
        y_pred = model.predict(X_test[feats])
        score_mse = mse(X_test['total_return'], y_pred)
        scores_mse.append(score_mse)
        print('mse')
        print(score_mse)
        draw_importance_features(model, X[feats], plot_type='gain')

    print('\n\nsummary')
    print('mse')
    print(sum(scores_mse) / len(scores_mse))


# In[8]:


save_results = False
if save_results:
    model = xgb.XGBRegressor(max_depth=5, n_estimators=100, learning_rate=0.3, seed=0, objective='reg:squarederror')
    train['total_return'] = train['price_total'] * train['is_canceled']
    model.fit(train[feats], train['total_return'])
    test['total_return'] = model.predict(test[feats])
    result = test.groupby('invoice')['total_return'].agg(np.sum).to_frame().reset_index()
    result[ ['invoice', 'total_return'] ].to_csv('../output/xgb_and_cnt_features_regression.csv', index=False)
    result[ ['invoice', 'total_return'] ]


# In[ ]:




