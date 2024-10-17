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
                    "k-Neighbour Regressor":KNeighborsRegressor(),
                    "Lasso":Lasso(),
                    "Ridge":Ridge(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoostRegressor":CatBoostRegressor(verbose=False)
                    
                }
                   
            model_report:dict=evaluate_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,
                                             models=models) 
            
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
            