

import pandas as pd

get_ipython().run_line_magic('run', 'custom_tools.py')



train = pd.read_hdf('../input/train_online_retail.h5')
test = pd.read_hdf('../input/test_online_retail.h5')



orders_train, orders_test = prepare_train_and_test(train, test)



feats = get_features(orders_train)
temp_feats = feats.copy()
temp_feats.append('total_return')
X_train = orders_train[temp_feats]
y_train = orders_train['is_canceled']
X_test = orders_test[feats]



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
pd.set_option("display.max_rows", 340)

bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X_train[feats],y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train[feats].columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(100,'Score'))  #print 10 best features



feats = get_features(orders_train)
X_train = orders_train[feats]
y_train = orders_train['total_return']
X_test = orders_test[feats]

import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse, f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
models = [
    xgb.XGBRegressor(max_depth=7, n_estimators=150, learning_rate=0.2, seed=0)
    #RandomForestClassifier(n_estimators=150, max_depth=7, max_features='auto')
]
def draw_importance_features(model, df, plot_type='gain'):
    fscore = model.get_booster().get_score(importance_type=plot_type) #cover, gain, weight
    maps_name = dict([ ("{0}".format(col), col) for i, col in enumerate(df.columns)])
    impdf = pd.DataFrame([ {'feature': maps_name[k], 'importance': fscore[k]} for k in fscore ])
    impdf = impdf.sort_values(by='importance', ascending=False).reset_index(drop=True)
    impdf['importance'] /= impdf['importance'].sum()
    impdf.index = impdf['feature']
    print(impdf.plot(kind='bar', title=f'{plot_type} - Importance Features', figsize=(15, 14)))

def model_train_predict(model, X, y, feats):
    cv = KFold(n_splits=2, random_state=0, shuffle=True)
    scores_mse = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx][feats], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx][feats])
        score_mse = mse(y.iloc[test_idx], y_pred)
        scores_mse.append(score_mse)
        #draw_importance_features(model, X.iloc[train_idx][feats], plot_type='gain')
        #draw_importance_features(model, X.iloc[train_idx][feats], plot_type='cover')
        #draw_importance_features(model, X.iloc[train_idx][feats], plot_type='weight')
    return {'mse':scores_mse}

for model in models:
    scores = model_train_predict(model, X_train, y_train, feats)
    print(model)
    for key, item in scores.items():
        print(key)
        print(item)

    for key, item in scores.items():
        print(key)
        print(sum(item) / len(item))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse
model = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.3, seed=0)

model.fit(X_train[feats], y_train)
y_pred = model.predict(X_test)
y_pred = y_pred*X_test['price_total']

mse(y_test_w['total_return'], y_pred)



orders_to_save = orders_test.copy()
orders_to_save['total_return'] = y_pred
orders_to_save



orders_to_save = orders_to_save.reset_index()
orders_to_save[ ['invoice', 'total_return'] ].to_csv('../output/first_method.csv', index=False)



orders_to_save[ ['invoice', 'total_return'] ]





