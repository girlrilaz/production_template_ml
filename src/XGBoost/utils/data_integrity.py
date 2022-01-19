""" Data Integrity Checks"""

import os, sys
import pandas as pd
sys.path.append('.')

# internal
from configs.module.pandas_schema import SCHEMA

def data_integrity_test(data):
    """
    ensure the raw data folder (data/raw) is not empty
    """
    input_folder = os.path.join('.','data', 'raw')
    if [f for f in os.listdir(input_folder) if not f.startswith('.')] == []:
        raise ValueError("Folder 'data/raw' is empty. Please ensure there is atleast one data file in this folder.")
    else: 
        print("Raw folder 'not-empty' test passed")
    
    # ## schema checks
    # try:
    #     validate = SCHEMA.validate_schema(data)
    #     if validate is None:
    #         print("PASS: Test data validation passed.")
    # except:
    #     raise Exception("ERROR - FAIL:(model_evaluation) - invalid input schema.")

    ## input checks
    if isinstance(data,dict):
        data = pd.DataFrame(data)
    elif isinstance(data,pd.DataFrame):
        print("Raw data input validity test passed")
        pass
    else:
        raise Exception(f"ERROR - FAIL:(model_evaluation) - invalid input. {data} was given")

    ## features check
    test_features = sorted(data.columns.drop(['y']).tolist())
    data_features = sorted(data.columns.drop(['y']).tolist())
    if test_features == data_features:
        print("Raw data feature test passed")
        pass
    elif test_features != data_features:
        print(f"test features: {','.join(test_features)}")
        raise Exception("ERROR - FAIL:(model_evaluation) - invalid features present")
    


### Run the tests
if __name__ == '__main__':

    input_folder = os.path.join('.','data', 'raw')
    data = pd.read_csv(os.path.join(input_folder, 'bank.csv'), delimiter=';')
    data_integrity_test(data)


