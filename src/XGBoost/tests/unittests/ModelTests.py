#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
import pickle
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from models.XGboost1 import *
from configs.module.config import CFG
from models.XGboost1 import XGB

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """
        # config_in = config.CFG #get CFG from config.py
    
        # train the model
        model = XGB(CFG)
        model.load_data(subset=True)
        model.build()
        model.train(subset=True)
        self.assertTrue(os.path.exists(os.path.join("models", "saved_models", "XGBoost", "1", "XGBoost_1.0.0-subset.pickle")))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        saved_model = os.path.join('./models/saved_models/XGBoost/1/XGBoost_1.0.0-subset.pickle')
        with open(saved_model, 'rb') as w:
            model = pickle.load(w)
            w.close()

        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

    # def test_03_predict(self):
    #     """
    #     test the predict function input
    #     """

    #     ## load model first
    #     model = model_load(test=True)
    
    #     ## ensure that a list can be passed
    #     query = pd.DataFrame({'sepal_length': [5.1, 6.4],
    #                         'sepal_width': [3.5, 3.2],
    #                         'petal_length': [1.4, 4.5],
    #                         'petal_width': [0.2, 1.5]
    #     })

    #     result = model_predict(query, model, test=True)
    #     y_pred = result['y_pred']
    #     self.assertTrue(y_pred[0] in ['setosa','versicolor', 'virginica'])

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
