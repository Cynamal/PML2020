#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import gc
import random
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
get_ipython().run_line_magic('run', 'custom_tools_2.py')


# ## Loading data

# In[2]:


try:
    if not df_all.empty:
        print('data already imported')
except:
    train_h5 = pd.read_hdf('../input/train_online_retail.h5')
    test_h5 = pd.read_hdf('../input/test_online_retail.h5')
finally:
    df_all = pd.concat([train_h5, test_h5], sort=False)


# ## Feature engineering

# In[4]:


prepared_df = prepare_features(df_all)
train = prepared_df[~prepared_df['is_canceled'].isnull()].copy().reset_index(drop=True)
test = prepared_df[prepared_df['is_canceled'].isnull()].copy().reset_index(drop=True)
train = prepare_additional_features(train)
test = prepare_additional_features(test)

feats = get_feats(train)
X_train = train[feats]
y_train = train['is_canceled'].astype(bool)


# ## Columns importance

# In[6]:


bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X_train[feats],y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train[feats].columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(120,'Score'))  #print 10 best features


# ## Validation

# In[7]:


models = [
    #RandomForestClassifier(random_state=0, n_estimators=50, max_depth=10),
    xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.3, seed=0)
]
def draw_importance_features(model, df, plot_type='gain'):
    fscore = model.get_booster().get_score(importance_type=plot_type) #cover, gain, weight
    maps_name = dict([ ("{0}".format(col), col) for i, col in enumerate(df.columns)])
    impdf = pd.DataFrame([ {'feature': maps_name[k], 'importance': fscore[k]} for k in fscore ])
    impdf = impdf.sort_values(by='importance', ascending=False).reset_index(drop=True)
    impdf['importance'] /= impdf['importance'].sum()
    impdf.index = impdf['feature']
    print(impdf.plot(kind='bar', title=f'{plot_type} - Importance Features', figsize=(15, 14)))

def model_train_predict(model, train, feats):
    np.random.seed(0)
    invoice_ids = train.invoice.unique()
    np.random.shuffle(invoice_ids)
    n = int(len(invoice_ids) * 0.1)
    invoice_light = set(invoice_ids[:n])
    train_light = train[ train.invoice.map(lambda x: x in invoice_light) ].reset_index(drop=True)
    
    cancelled = train_light[train_light['is_canceled'] == True]['invoice'].unique()
    not_cancelled = train_light[train_light['is_canceled'] == False]['invoice'].unique()
    scores_f1, scores_recall, scores_precision, scores_mse = [], [], [], []
    X = train_light[feats]
    y = train_light['is_canceled'].astype(bool)
    for i in range(3): 
        test_invoice_true = list(np.random.choice(list(cancelled), 124, replace=False)) 
        test_invoice_false = list(np.random.choice(list(not_cancelled), 680, replace=False))
        test_invoice = test_invoice_true + test_invoice_false
        train_idx = train_light[~train_light['invoice'].isin(test_invoice)].index
        test_idx = train_light[train_light['invoice'].isin(test_invoice)].index
        model.fit(X.iloc[train_idx][[feat for feat in feats if feat != 'price_total']], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx][[feat for feat in feats if feat != 'price_total']])
        #calculate scores
        score_f1 = f1_score(y.iloc[test_idx], y_pred)
        score_recall = recall_score(y.iloc[test_idx], y_pred)
        score_precision = precision_score(y.iloc[test_idx], y_pred)
        #mse score preparation
        real_values = X.iloc[test_idx].copy()
        real_values['is_canceled'] = y.iloc[test_idx]
        real_values['total_return'] = real_values['price_total'] * real_values['is_canceled']
        real_values = real_values.groupby('invoice')['total_return'].agg(np.sum).to_frame().reset_index()
        validation = X.iloc[test_idx].copy()
        validation['is_canceled'] = y_pred
        validation['total_return'] = validation['price_total'] * validation['is_canceled']
        validation = validation.groupby('invoice')['total_return'].agg(np.sum).to_frame().reset_index()
        score_mse = mse(real_values['total_return'], validation['total_return'])
        
        #append scores
        scores_f1.append(score_f1)
        scores_recall.append(score_recall)
        scores_precision.append(score_precision)
        scores_mse.append(score_mse)
        
        #feature importance
        draw_importance_features(model, X.iloc[train_idx][[feat for feat in feats if feat != 'price_total']], plot_type='gain')
        #draw_importance_features(model, X.iloc[train_idx][[feat for feat in feats if feat != 'price_total']], plot_type='cover')
        #draw_importance_features(model, X.iloc[train_idx][[feat for feat in feats if feat != 'price_total']], plot_type='weight')
    return {'f1':scores_f1, 'recall':scores_recall, 'precision':scores_precision, 'mse':scores_mse}


# In[10]:


for model in models:
    scores = model_train_predict(model, train, feats)
    for key, item in scores.items():
        print(key)
        print(item)

    for key, item in scores.items():
        print(key)
        print(sum(item) / len(item))


# ## Getting predictions

# In[11]:


model = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.3, seed=0)
model.fit(train[[feat for feat in feats if feat != 'price_total']], train['is_canceled'])
test['is_canceled'] = model.predict(test[[feat for feat in feats if feat != 'price_total']])
test['total_return'] = test['price_total'] * test['is_canceled']
result = test.groupby('invoice')['total_return'].agg(np.sum).to_frame().reset_index()
result[ ['invoice', 'total_return'] ].to_csv('../output/xgb_and_cnt_features.csv', index=False)


# In[ ]:




