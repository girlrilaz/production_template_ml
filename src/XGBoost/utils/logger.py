"""
basic test procedure for logger.py
"""

import logging.config
import yaml
import time,os,re,csv,sys,uuid,joblib
from datetime import date

today = date.today()
day_folder = f"{today.year}-{today.month:02d}-{today.day}"

if not os.path.exists(os.path.join(".","logs",day_folder)):
    os.mkdir(os.path.join("logs", day_folder))

with open('configs/yaml/logging_config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logging.captureWarnings(True)

def get_logger(name: str):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    return logger

def update_train_log(data_shape, runtime, model_version, model_version_note, best_params, subset=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    day_folder = f"{today.year}-{today.month:02d}-{today.day}"

    if subset:
        logfile = os.path.join("logs", day_folder, f"model-train-subset.log")
    else:
        logfile = os.path.join("logs", day_folder,f"model-train-{today.year}-{today.month:02d}-{today.day}.log")
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','x_shape','model_version',
              'model_version_note','runtime', 'model_params']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),data_shape,
                            model_version,model_version_note,runtime, best_params])
        writer.writerow(to_write)

def update_evaluation_log(eval_metrics,runtime,model_version):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    day_folder = f"{today.year}-{today.month:02d}-{today.day}"
 
    logfile = os.path.join("logs", day_folder ,f"model-eval-{today.year}-{today.month:02d}-{today.day}.log")
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp','model_version','runtime','eval_metrics']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(), model_version,runtime, eval_metrics])
        writer.writerow(to_write)

# if __name__ == "__main__":

#     """
#     basic test procedure for logger.py
#     """

#     ## train logger
#     update_train_log(str((100,10)), "00:00:01", "1.0.0", "version_note", "{'learning_rate':0.05}", subset=True)

#     ## predict logger
#     update_evaluation_log('{"accuracy": 0.91, "roc_auc": 0.88}',"00:00:01","1.0.0")