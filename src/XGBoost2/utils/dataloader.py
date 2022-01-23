# -*- coding: utf-8 -*-
"""Data Loader"""

# standard library
import sys
import pandas as pd

# external
#import jsonschema
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# internal
from configs.module.pandas_schema import SCHEMA

sys.path.append('.')

class DataLoader:

    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        return pd.read_csv(data_config.path, delimiter=';')

    @staticmethod
    def validate_schema(data_point):
        """Data schema validation"""
        SCHEMA.validate(data_point)
        #jsonschema.validate({'data':data_point.tolist()},SCHEMA)

    @staticmethod
    def split_feature_target(data_point, y_attribute):
        """Split features (X) and target (Y)"""
        X = data_point.drop(y_attribute, axis = 1)
        y = data_point[y_attribute]
        return X, y

    @staticmethod
    def preprocess_data(dataset, test_size, random_state):
        """ Preprocess and splits into training and test"""
        return train_test_split(dataset, test_size=test_size, random_state=random_state)

    @staticmethod
    def feature_pipeline(numerical_features, categorical_features): #numerical_features, categorical_features
        """Loads and preprocess a datapoint with pipeline"""

        ## preprocessing pipeline
        # numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                            ('scaler', StandardScaler())])

        # categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                                                

        X_pipeline = ColumnTransformer(transformers=[('num', numerical_pipeline, numerical_features),
                                                    ('cat', categorical_pipeline, categorical_features)])

        return X_pipeline

    @staticmethod
    def target_pipeline(target_features):
        """Loads and preprocess a datapoint with pipeline"""

        y_pipeline = Pipeline(steps=[('label_enc', MyLabelEncoder())])

        return y_pipeline


## custom LabelEncoder as the org doesn't work in pipeline - this is a workaround
class MyLabelEncoder(LabelEncoder):
    """
     custom LabelEncoder as the org doesn't work with the pipeline
    """
    def fit_transform(self, X, y=None):
        return super(MyLabelEncoder, self).fit_transform(X)
    
    def fit(self, X, y=None):
        return super(MyLabelEncoder, self).fit(X.values.ravel())
    
    def transform(self, X):
        return super(MyLabelEncoder, self).transform(X.values.ravel())

