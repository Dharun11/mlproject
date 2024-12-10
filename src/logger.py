import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # this is the log folder name nd log file name both are same
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)  # this will create a folder name "logs" in cwd here we declare the path 
os.makedirs(logs_path,exist_ok=True)   #creating the logs folder

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)   # this is the entire path inside logs folder , it will have a folder name in LOG_FILE format 
# inside it will create name 

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s -%(message)s",
    level=logging.INFO,
)


if __name__=="__main__":
    logging.info("This is a log message LOGGING HAS STARTED")

