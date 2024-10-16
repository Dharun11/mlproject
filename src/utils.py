import sys
import os
import numpy as np
import dill
from src.exception import CustomException

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