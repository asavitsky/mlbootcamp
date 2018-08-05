import pandas as pd
import json
import xgboost as xgb
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz,csr_matrix,hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def avg_ranks(rk1,rk2,alpha=0.5):
    rks=pd.DataFrame(pd.Series(rk1).rank())
    rks.columns=['rank_lr']
    rks['rank_xgb']=pd.Series(rk2).rank()
    rks['final_pred']=rks.apply(lambda x: (alpha*x[0]+(1-alpha)*x[1]),axis=1)
    rks['final_pred']=rks['final_pred']/rks['final_pred'].max()
    return rks['final_pred'].values

def train_test_ids():
    y_train=pd.read_csv('mlboot_train_answers.tsv',sep='\t')
    test_id=pd.read_csv('mlboot_test.tsv',sep='\t')
    y_train.columns=['id','target']
    test_id.columns=['id']
    return y_train,test_id

def time_features(data):
    fts_days=data.groupby('id')['days'].nunique().reset_index()
    fts_days['all_days']=data.groupby('id')['days'].count().values
    fts_days['vis_per_day']=fts_days['all_days']/fts_days['days']
    fts_days['len_in_days']=data.groupby('id')['days'].max().values-data.groupby('id')['days'].min().values
    fts_days['mean_time_distance']=fts_days['len_in_days']/fts_days['days']
    fts_days['min_day']=data.groupby('id')['days'].min().values
    fts_days['max_day']=data.groupby('id')['days'].max().values
    return fts_days

def cat1_features(data):
    cat1_fts=data.groupby('id')['cat1'].mean().reset_index()
    cat1_fts['last_cat1']=data.groupby('id')[['days','cat1']].apply(lambda x: x.sort_values('days').head(1)).values[:,1]
    cat1_val=pd.DataFrame(data.groupby('id')['cat1'].value_counts())
    cat1_val.columns=['cat1_ct']
    cat1_val=cat1_val.reset_index(level='cat1')
    cat1_val=cat1_val.pivot(columns='cat1')
    cat1_val.columns=['cat'+str(x) for x in range(0,6)]
    cat1_val=cat1_val.fillna(0).reset_index()
    cat1_fts=pd.merge(cat1_fts,cat1_val,on='id',how='left')
    return cat1_fts


def prepare_ct1(data):
    data['ct1']=data['ct1'].apply(lambda x: collections.Counter(json.loads(x)))
    id_ct1=data.groupby('id')['ct1'].sum()
    id_ct1=id_ct1.reset_index()
    id_ct1['ct1']=id_ct1['ct1'].apply(lambda x: dict(x))
    id_ct1.to_csv('id_ct1.csv',index=False)
    def renew(x):
        s=''
        for k in x.keys():
            s+=x[k]*(k+' ')
        return s.strip()

    id_ct1['ct1']=id_ct1['ct1'].apply(lambda x: renew(x))
    id_ct1['ct1'] = id_ct1['ct1'].fillna('')
    return id_ct1

def prepare_ct2(data):
    def renew_counter2(x):
        s=''
        for k in x.keys():
            s+=x[k]*(k+' ')
        return s

    data['ct2']=data['ct2'].apply(lambda x: renew_counter2(json.loads(x)))
    id_ct2=data.groupby('id')['ct2'].sum()
    id_ct2=id_ct2.reset_index()
    id_ct2['ct2'] = id_ct2['ct2'].fillna('')
    return id_ct2


def log_reg(x,y,test):
    model_lr=LogisticRegression(random_state=111,C=0.1)
    model_lr.fit(x,y)
    prediction=model_lr.predict_proba(test)[:,1]
    return prediction

def xgb_model(x,y,test):
    dtrain = xgb.DMatrix(x,label=y)
    dtest = xgb.DMatrix(test)
    params={'max_depth': 3, 'eta': 0.05, 'objective': 'binary:logistic','seed':111,'eval_metric':'auc','subsample':0.8,'colsample_bytree':0.8}
    bst = xgb.train(params, dtrain, 350)
    prediction=bst.predict(dtest)
    return prediction



def main():
    features = ['days', 'all_days', 'vis_per_day', 'len_in_days', 'mean_time_distance', 'min_day', 'max_day', 'cat1_x',
                'last_cat1', 'cat0', 'cat2', 'cat3', 'cat4', 'cat5']
    data = pd.read_csv('mlboot_data.tsv', header=None, sep='\t', names=['id', 'cat1', 'ct1', 'ct2', 'ct3', 'days'],
                       usecols=[0,1,2,3,5])

    y_train, test_id=train_test_ids()
    id_ct1=prepare_ct1(data[['id','ct1']])
    id_ct2=prepare_ct2(data[['id','ct2']])

    train = pd.merge(y_train, id_ct1, on='id', how='left')
    train=pd.merge(train,id_ct2,on='id',how='left')

    test = pd.merge(test_id, id_ct1, on='id', how='left')
    test=pd.merge(test,id_ct2,on='id',how='left')
    target = train['target'].values

    fts_days=time_features(data[['id','days']])
    train = pd.merge(train, fts_days, on='id', how='left')
    test = pd.merge(test, fts_days, on='id', how='left')

    cat1_fts=cat1_features(data[['id','cat1']])
    train = pd.merge(train, cat1_fts, on='id', how='left')
    test = pd.merge(test, cat1_fts, on='id', how='left')

    'Tfidf on counter1'
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=10, max_features=10000)
    train_tfidf = tfidf_vectorizer.fit_transform(train['ct1'])
    test_tfidf = tfidf_vectorizer.transform(test['ct1'])

    sc = StandardScaler()
    train[features] = sc.fit_transform(train[features])
    test[features] = sc.transform(test[features])

    train_tfidf = hstack([train_tfidf, csr_matrix(train[features].values)])
    test_tfidf = hstack([test_tfidf, csr_matrix(test[features].values)])

    'Prediction on subset1'
    prediction_lr1 = log_reg(train_tfidf, target, test_tfidf)
    xgb_prediction1 = xgb_model(train_tfidf, target, test_tfidf)
    pred1 = avg_ranks(prediction_lr1, xgb_prediction1, 0.3)

    'Tfidf on counter2'
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=10, max_features=10000)
    train_tfidf = tfidf_vectorizer.fit_transform(train['ct2'])
    test_tfidf = tfidf_vectorizer.transform(test['ct2'])

    sc = StandardScaler()
    train[features] = sc.fit_transform(train[features])
    test[features] = sc.transform(test[features])
    train_tfidf = hstack([train_tfidf, csr_matrix(train[features].values)])
    test_tfidf = hstack([test_tfidf, csr_matrix(test[features].values)])

    'Prediction on subset2'
    prediction_lr2 = log_reg(train_tfidf, target, test_tfidf)
    xgb_prediction2 = xgb_model(train_tfidf, target, test_tfidf)
    pred2 = avg_ranks(prediction_lr2, xgb_prediction2, 0.5)
    'Blending'
    pred=avg_ranks(pred1,pred2,0.4)

    np.savetxt('pred_test.csv',pred)

if(__name__ == "__main__"):
    main()