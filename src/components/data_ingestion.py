import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # you will be able to directly define your  class variables 

from src.components.data_transformation import DataTransformation,DataTransformationConfig
'''
when the class only haves the variables then we can use dataclass or else it is better to use init if it contains
functions and var
'''
# we are creating a class that gets all the inputs required 

@dataclass
class DataIngestionConfig:
    """This class contains configuration for data ingestion."""
    train_data_path:str=os.path.join('artifacts','train.csv')  # train-data-path is an input and we are initializing where its output to be stored 
    # we said that the o/p will be stored on "artifacts folder" under the name "train.csv"\
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')


#our main class
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # here this var will get all 3 variables from the above class

    '''
    this function is to get the data from external KB such as sqlite , mongo db
    but for now we will get data asusual
    '''
    def initiate_data_ingestion(self):
        logging.info("Entered into the data ingestion component")
        try:
            df=pd.read_csv('Notebook\data\stud.csv')
            logging.info("Read the data as a dataframe")
            
            #now creating the directories for the i/p var
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            #storing the raw data to desired file path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            #storing the train and test data to desried folder
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("ingestion of the data has been completed")
            
            return(
                # these 2 data paths are needed for data_transformation
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
            
            
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    obj2=DataTransformation()
    obj2.intiate_data_transformation(train_data,test_data)