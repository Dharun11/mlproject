import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


from xgboost import XGBRegressor

from src.utils import save_obj,evaluate_model

'''
Declaring a  required input w.r.t to this file

a var for create a pkl after creating our model
'''

@dataclass
class ModelTrainerConfig:
    model_trained_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting the train and test data")
            
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1], #removing the last column i.e target feature,
                train_arr[:,-1],   # taking the last column i.e target feature
                test_arr[:,:-1],   #removing the last column i.e target feature,
                test_arr[:,-1]     # taking the last column i.e target feature
                
            )
            
            models= {
                    "LinearRegression" : LinearRegression(),
                    "DecisionTreeRegressor" : DecisionTreeRegressor(),
                    "GradientBoostRegressor": GradientBoostingRegressor(),
                    "RandomForestRegressor" : RandomForestRegressor(),
                    "AdaBoostRegressor": AdaBoostRegressor(),
                    "KNeighborRegressor":KNeighborsRegressor(),
                    "Lasso":Lasso(),
                    "Ridge":Ridge(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoostRegressor":CatBoostRegressor(verbose=False)
                    
                }
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Lasso": {
                    'alpha': [0.1, 0.01, 0.001, 1, 10, 100],
                    'max_iter': [1000, 2000, 3000],
                    'tol': [0.0001, 0.00001, 0.000001]
                },

                "Ridge": {
                    'alpha': [0.1, 0.01, 0.001, 1, 10, 100],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
                    'max_iter': [None, 1000, 2000]
                },

                "KNeighborRegressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
                   
            model_report:dict=evaluate_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,
                                             models=models,param=params) 
            
            ## to get the best model from dict
            best_model_score=max(sorted(model_report.values()))
            
            #to get the best model name from dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("There is no best model")
            logging.info("We found the best model on both training and test data")
            
            save_obj(file_path=self.model_trainer_config.model_trained_file_path,obj=best_model)
            
            logging.info("Best Model file has been created and saved")
            
            predicted=best_model.predict(x_test)
            best_r2_score=r2_score(y_test,predicted)
            
            return best_r2_score,best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)
            