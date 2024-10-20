import sys
import os
import numpy as np
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, 'wb') as output:
            dill.dump(obj,output)
            
    except Exception as e:
        raise CustomException(e,sys)
'''
dill is used to create .pkl file
'''        

def evaluate_model(x_train,x_test,y_train,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            
            model.fit(x_train,y_train)
            
            y_train_predicted=model.predict(x_train)
            y_test_predicted=model.predict(x_test)
            
            modeltrain_r2_score=r2_score(y_train,y_train_predicted)
            modeltest_r2_score=r2_score(y_test,y_test_predicted)
            
            report[list(models.keys())[i]]= modeltest_r2_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)