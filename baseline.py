import os
import pandas as pd
from pandas.tseries.offsets import DateOffset
from lightgbm import LGBMRegressor
import numpy as np
import pickle

pd.options.mode.chained_assignment = None

#scikit-learn==1.2.1
#numpy==1.23.5
#tqdm==4.65.0
#pandas==1.5.3
#lightgbm==4.1.0

BEST_PARAM = [[3, 5.486299354001131, {'n_estimators': 120, 'max_depth': 30, 'num_leaves': 45, 'learning_rate': 0.031}]
              , [4, 6.336036861875657, {'n_estimators': 120, 'max_depth': 40, 'num_leaves': 50, 'learning_rate': 0.031}]
              , [5, 6.624264784072142, {'n_estimators': 180, 'max_depth': 55, 'num_leaves': 20, 'learning_rate': 0.061}]
              , [6, 6.6113787115191585, {'n_estimators': 170, 'max_depth': 5, 'num_leaves': 75, 'learning_rate': 0.031}]
              , [7, 6.711997672986595, {'n_estimators': 120, 'max_depth': 40, 'num_leaves': 15, 'learning_rate': 0.091}]
              , [8, 6.825391111034213, {'n_estimators': 70, 'max_depth': 95, 'num_leaves': 5, 'learning_rate': 0.121}]
              , [9, 7.065619482796654, {'n_estimators': 450, 'max_depth': 5, 'num_leaves': 25, 'learning_rate': 0.011}]
              , [10, 7.315422658300333, {'n_estimators': 100, 'max_depth': 40, 'num_leaves': 10, 'learning_rate': 0.131}]]
DIR_DF = 'data'
FILE_TRAIN = 'train.csv'
FILE_PEDIGREE = 'pedigree.csv'
FILE_X_TEST_PUBLIC = 'X_test_public.csv'
FILE_SUBMISSION = 'submission.csv'
PKL_FILE = 'pkl_model.pkl'
RANDOM_STATE = 1
TEST_SIZE = .25

columns_features = ['lactation','farm','calving_year_month','age_year_month','milk_yield_1','milk_yield_2']

def fit():
    
    if os.path.exists(PKL_FILE):
        with open(PKL_FILE, 'rb') as file: 
            ret_model = pickle.load(file)
    else:
        df2 = pd.read_csv(os.path.join(DIR_DF,FILE_TRAIN))
        df2[['calving_date','birth_date']] = df2[['calving_date','birth_date']].astype("datetime64[ns]")
        df2['calving_year_month'] = df2['calving_date'].dt.month
        df2['age_year_month'] = (df2['calving_date'].dt.to_period("M").astype('int64') 
                                 - df2['birth_date'].dt.to_period("M").astype('int64'))
        
        columns_categorical_feature = ['lactation','farm','calving_year_month']
        
        ret_model=[]
        for i in range(3,11):
            df2_temp = df2[columns_features]
            df2_temp[f'milk_yield_{i}'] = df2[f'milk_yield_{i}'] 
            df2_temp = df2_temp.dropna()
            df2_temp = df2_temp.drop_duplicates()            
            params = BEST_PARAM[i-3][2]
            model_lgm = LGBMRegressor(**params)
            model_lgm.fit(df2_temp[columns_features],df2_temp[f'milk_yield_{i}'],eval_metric='rmse'
                         ,categorical_feature=columns_categorical_feature)
            ret_model.append(model_lgm)
        with open(PKL_FILE, 'wb') as file: 
            pickle.dump(ret_model, file)
            
    #print(1)
    return ret_model

def predict(model, test_dataset_path: str) -> pd.DataFrame:
       
    test_dataset = pd.read_csv(test_dataset_path)
    test_dataset[['calving_date','birth_date']] = test_dataset[['calving_date','birth_date']].astype("datetime64[ns]")
    test_dataset['calving_year_month'] = test_dataset['calving_date'].dt.month
    test_dataset['age_year_month'] = (test_dataset['calving_date'].dt.to_period("M").astype('int64') 
                             - test_dataset['birth_date'].dt.to_period("M").astype('int64'))
    ret_df = pd.DataFrame(test_dataset[['animal_id', 'lactation']])
    for i in range(3,11):
        ret_df[f'milk_yield_{i}'] = model[i-3].predict(test_dataset[columns_features])
        
    return ret_df

if __name__ == '__main__':
    _model = fit()
    
    _submission = predict(_model, os.path.join(DIR_DF, FILE_X_TEST_PUBLIC))
    _submission.to_csv(os.path.join(DIR_DF, FILE_SUBMISSION), sep=',', index=False)
