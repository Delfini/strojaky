import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import random

lmtzr = WordNetLemmatizer()
random.seed(23)
cachedStopWords = stopwords.words("english")

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

def str_lemmatize(str1):
    str1 = str(str1)
    str1 = str1.lower()
    str1 = str1.replace(" in.","in.")
    str1 = str1.replace(" inch","in.")
    str1 = str1.replace("inch","in.")
    str1 = str1.replace(" in ","in. ")
    str1 = (" ").join([lmtzr.lemmatize(z) for z in str1.split(" ")])
    return str1

def transform_titles(documents):
    global sklearn_titles, sklearn_tfidf_title
    sklearn_tfidf_title = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words='english')
    sklearn_titles = sklearn_tfidf_title.fit_transform(documents)

def cosine_distance_titles(str1):
    global item, sklearn_titles, sklearn_tfidf_title
    str1_matrix = sklearn_tfidf_title.transform([str1])
    item = item + 1
    return cosine_similarity(str1_matrix[0:1], sklearn_titles[item-1:item]).item(0,0)

def transform_descriptions(documents):
    global sklearn_descriptions, sklearn_tfidf_desc
    sklearn_tfidf_desc = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words='english')
    sklearn_descriptions = sklearn_tfidf_desc.fit_transform(documents)

def cosine_distance_desc(str1):
    global item, sklearn_descriptions, sklearn_tfidf_desc
    str1_matrix = sklearn_tfidf_desc.transform([str1])
    item = item + 1
    return cosine_similarity(str1_matrix[0:1], sklearn_descriptions[item-1:item]).item(0,0)

def transform_attributes(documents):
    global sklearn_attributes, sklearn_tfidf_attr
    sklearn_tfidf_attr = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, stop_words='english')
    sklearn_attributes = sklearn_tfidf_attr.fit_transform(documents)

def cosine_distance_attr(str1):
    global item, sklearn_attributes, sklearn_tfidf_attr
    str1_matrix = sklearn_tfidf_attr.transform([str1])
    item = item + 1
    return cosine_similarity(str1_matrix[0:1], sklearn_attributes[item-1:item]).item(0,0)


df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')
df_all['value'] = df_all['value'].fillna("tada")
#lemmatizing
df_all['search_term'] = df_all['search_term'].map(lambda x:str_lemmatize(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_lemmatize(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_lemmatize(x))
df_all['value'] = df_all['value'].map(lambda x:str_lemmatize(x))

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']+"\t"+df_all['value']

transform_titles(df_all['product_title'].tolist())
item = 0
df_all['query_in_title_dist'] = df_all['product_info'].map(lambda x:cosine_distance_titles(x.split('\t')[0]))
transform_descriptions(df_all['product_description'].tolist())
item = 0
df_all['query_in_desc_dist'] = df_all['product_info'].map(lambda x:cosine_distance_desc(x.split('\t')[0]))
transform_attributes(df_all['value'].tolist())
item = 0
df_all['query_in_attr_dist'] = df_all['product_info'].map(lambda x:cosine_distance_attr(x.split('\t')[0]))

df_all = df_all.drop(['search_term','product_title','product_description','product_info', 'value'],axis=1)
#Data Frame description
print(df_all.describe())

df_train = df_all.iloc[:num_train]
#Data Frame description
print(df_train.describe())
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
id_train = df_train['id']
y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'product_uid','relevance'],axis=1).values
X_test = df_test.drop(['id', 'product_uid','relevance'],axis=1).values
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
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('rfr_cos.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))
