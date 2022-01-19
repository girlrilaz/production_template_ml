#!/usr/bin/env python
"""
model tests
"""

import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
from datetime import date
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from utils.logger import update_train_log, update_evaluation_log #, update_processing_log

class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        ensure log file is created
        """

        log_test_date = date.today() #"2022-1-15" #date.today()

        log_file = os.path.join("logs", f"{log_test_date}", "model-train-subset.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        data_shape = (100,10)
        best_params = {'learning_rate':0.05}
        runtime = "00:00:01"
        model_version = "1.0.0"
        model_version_note = "test model"
        
        update_train_log(data_shape, runtime, model_version, model_version_note, best_params, subset=True)

        self.assertTrue(os.path.exists(log_file))
        
    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """
        log_test_date = date.today() #"2022-1-15" #date.today()

        log_file = os.path.join("logs", f"{log_test_date}", "model-train-subset.log")
        
        ## update the log
        data_shape = (100,10)
        best_params = {'learning_rate':0.05}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"
        
        update_train_log(data_shape, runtime, model_version, model_version_note, best_params, subset=True)

        df = pd.read_csv(log_file)
        logged_best_params = [literal_eval(i) for i in df['model_params'].copy()][-1]
        self.assertEqual(best_params, logged_best_params)

    def test_03_evaluation(self):
        """
        ensure log file is created
        """
        log_test_date = date.today() #"2022-1-15" #date.today()
        log_file = os.path.join("logs" , f"{log_test_date}" , f"model-eval-{log_test_date}.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## update the log
        runtime = "00:00:02"
        model_version = 0.1
        eval_metrics = {"accuracy": 0.91, "roc_auc": 0.88}

        update_evaluation_log(eval_metrics, runtime, model_version)
        
        self.assertTrue(os.path.exists(log_file))
    
    def test_04_evaluation(self):
        """
        ensure that content can be retrieved from log file
        """
        log_test_date = date.today() #"2022-1-15" #date.today()
        log_file = os.path.join("logs" , f"{log_test_date}" ,f"model-eval-{log_test_date}.log")

        ## update the log
        runtime = "00:00:02"
        model_version = 0.1
        eval_metrics = {"accuracy": 0.91, "roc_auc": 0.88}

        update_evaluation_log(eval_metrics, runtime, model_version)

        df = pd.read_csv(log_file)
        logged_y_pred = [literal_eval(i) for i in df['eval_metrics'].copy()][-1]
        self.assertEqual(eval_metrics,logged_y_pred)

### Run the tests
if __name__ == '__main__':
    unittest.main()
      
