import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
import random

random.seed(23)

df_train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('./input/attributes.csv')
df_attr = df_attr.dropna()
df_attr = df_attr.reset_index(drop=True)
d = {}
for i in range(len(df_attr)):
    if str(int(df_attr.product_uid[i])) in d:
        d[str(int(df_attr.product_uid[i]))][1] += " " + str(df_attr['value'][i]).replace('\t'," ")
    else:
        d[str(int(df_attr.product_uid[i]))] = [int(df_attr.product_uid[i]),str(df_attr['value'][i])]
df_attr = pd.DataFrame.from_dict(d,orient='index')
df_attr.columns = ['product_uid','value']
df_pro_desc = pd.read_csv('./input/product_descriptions.csv')
num_train = df_train.shape[0]
print("Train set size", num_train)
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RSME  = make_scorer(fmean_squared_error, greater_is_better=False)

def str_stem(str1):
    str1 = str(str1)
    str1 = str1.lower()
    str1 = str1.replace(" in.","in.")
    str1 = str1.replace(" inch","in.")
    str1 = str1.replace("inch","in.")
    str1 = str1.replace(" in ","in. ")
    str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
    return str1

def str_common_word_tf(str1, str2):
    str1, str2 = str1.lower(), str2.lower()
    words, cnt = str1.split(), 0
    length = len(str2.split())
    for word in words:
        i = 0
        word_cnt = 0
        while i < len(str2):
            i = str2.find(word, i)
            if i == -1:
                break
            else:
                word_cnt += 1
                i += len(str1)
        cnt += float(word_cnt)/float(length)
    return float(cnt)/len(words) 

def str_whole_word_tf(str1, str2):
    str1, str2 = str1.lower().strip(), str2.lower().strip()
    n = len(str1.split())
    length = len(str2.split())
    if n > length:
      return 0
    cnt, suma = 0, 0
    i = 0
    while i < len(str2):
        i = str2.find(str1, i)
        if i == -1:
            break
        else:
            cnt += 1
            i += len(str1)
    return float(cnt)/float(length+1-n)

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')
df_all['value'] = df_all['value'].fillna("tada")
#stemming
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['value'] = df_all['value'].map(lambda x:str_stem(x))

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']+"\t"+df_all['value']

df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word_tf(x.split('\t')[0],x.split('\t')[1]))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word_tf(x.split('\t')[0],x.split('\t')[2]))
df_all['query_in_attribute_value'] = df_all['product_info'].map(lambda x:str_whole_word_tf(x.split('\t')[0], x.split('\t')[3]))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word_tf(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word_tf(x.split('\t')[0],x.split('\t')[2]))
df_all['word_in_attribute_value'] = df_all['product_info'].map(lambda x:str_common_word_tf(x.split('\t')[0],x.split('\t')[3]))

df_all = df_all.drop(['search_term','product_title','product_description','product_info', 'value'],axis=1)
df_train = df_all.iloc[:num_train]
#DataFrame description
print(df_train.describe())
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
id_train = df_train['id']
y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'product_uid','relevance'],axis=1).values
X_test = df_test.drop(['id', 'product_uid', 'relevance'],axis=1).values
print("--- Features Set: %s minutes ---" % ((time.time() - start_time)/60))
rfr = RandomForestRegressor()
clf = pipeline.Pipeline([('rfr', rfr)])
param_grid = {'rfr__n_estimators' : list(range(22,26,1)), 'rfr__max_depth': list(range(6,9,1))}
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RSME)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
print("Grid scores")
print(model.grid_scores_)

y_pred = model.predict(X_test)
print("Test data size", len(y_pred))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('rfr_tf1.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))
