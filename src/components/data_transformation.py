import sys
import os
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # for handling missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

#creating a config classs for getting the inputs for this file

@dataclass
class DataTransformationConfig:
    #this var is for storing the model in pkl format
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_features=['reading_score','writing_score']
            categorical_feature=[
                'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'
                ]
            
            #creating pipelines
            
            numeric_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean")),  #handling missing values using median
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ('Scaler',StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns onehotencoding completed")
            
        # using columntransformer we will combine these pipelines

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",numeric_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_feature)
                    
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def intiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("training and testing dataset reading completed")
            logging.info("Obtaining pre processing object")
            
            preprocess_obj=self.get_data_transformer_object()
            
            target_column_name=['math_score']
            numerical_features=['reading_score','writing_score']
            logging.info("Applying pre processing obj to train df and test df")
            
            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)   #x_train
            target_feature_train_df=train_df[target_column_name]     #y_train
            
            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)   #x_test
            target_feature_test_df=test_df[target_column_name]  #y_test
            
            input_feature_train_arr=preprocess_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocess_obj.transform(input_feature_test_df)
            
            
            # np.c_ is the column wise concatenation    
            '''
            # Two 1-D arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Column-wise concatenation
result = np.c_[a, b]
print(result)
            '''
            
        #output
            '''
            [[1 4]
 [2 5]
 [3 6]]
            '''
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)  # x_train,y_train
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)   #x_test,y_test
            ]
            
            logging.info("Saved preprocessed Objects........")
            
            #this function is written in utils.py used to create a folder and file for the pre processed obj
            save_obj(
                file_path=self.data_transform_config.preprocessor_obj_file_path,
                obj=preprocess_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        