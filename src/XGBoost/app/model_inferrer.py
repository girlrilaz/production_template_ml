
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

# from utils.visualize import display
from utils.config import Config
from configs.module.config import CFG
from utils.dataloader import DataLoader

class ModelInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG)
        self.numerical = self.config.data.numerical_att
        self.categorical = self.config.data.categorical_att
        self.target = self.config.train.target_att
        self.saved_path = './models/saved_models/XGBoost/1/XGBoost_1.0.0.pickle'
        self.model = pickle.load(open(self.saved_path, 'rb'))

    def preprocess(self, query):

        ## input checks
        if isinstance(query, dict):
            query = pd.DataFrame(query)
        elif isinstance(query, pd.DataFrame):
            pass
        else:
            raise Exception(f"ERROR - FAIL:(model_evaluation) - invalid input. {type(query)} was given")

        X = query.drop(['y'], axis=1)
        y = query[['y']]
        query_train = DataLoader().feature_pipeline(self.numerical, self.categorical).fit(X).transform(X) # get from model trainer log
   
        for n_samples in range(10, len(X), 20):

            subset_indices = X.sample(n=n_samples, replace=True)
            subset_query = DataLoader().feature_pipeline(self.numerical, self.categorical).fit(subset_indices).transform(subset_indices)
            if subset_query.shape[1] != query_train.shape[1]:
                continue
            else:
                    break

        # print(f'n_ : {n_samples}')
        # print(f'y_ : {subset_query.shape[1]}')

        return subset_query, subset_indices, y

    def infer(self, query):
  
        subset_query, subset_indices, y = self.preprocess(query)
        pred = self.model.predict(subset_query)
        pred = pred.tolist()

        # output in dictionary
        #prediction_dict = {'variable_input': subset_indices, 'actual_class': y, 'prediction_output':pred}

        # output in datadrame
        prediction_df = subset_indices
        prediction_df['actual_y'] = y
        prediction_df['pred_y'] = pred
        prediction_df['predicted_y'] = np.where(prediction_df['pred_y']==0, 'no', 'yes')
        prediction_df.drop('pred_y', axis = 1, inplace=True)

        return prediction_df


if __name__ == '__main__':

    # query = {'age': [30, 33],
    #         'job': ["unemployed", "services"],
    #         "marital": ["married", "married"],
    #         "education": ["primary", "secondary"],
    #         "default": ["no", "no"],
    #         "balance": [1787, 4789],
    #         "housing": ["no", "yes"],
    #         "loan": ["no", "yes"],
    #         "contact": ["cellular", "cellular"],
    #         "day": [19, 11],
    #         "month": ["oct", "may"],
    #         "duration": [79, 220],
    #         "campaign": [1, 1],
    #         "pdays": [-1, 339],
    #         "previous": [0, 4],
    #         "poutcome":["unknown", "failure"]
    # }

    df = pd.read_csv("./data/raw/bank.csv", delimiter=';')

    print(ModelInferrer().infer(df))
