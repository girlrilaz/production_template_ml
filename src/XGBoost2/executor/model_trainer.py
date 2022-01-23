# -*- coding: utf-8 -*-
"""Model Training Executor"""

# standard library
import os
import re
import time
import pickle

#external
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

#internal
from utils.logger import get_logger, update_train_log

LOG = get_logger('xgboost_training')

class XGBTrainer:
    """Model training executor class"""

    def __init__(self, model, name, folder, version, desc, X_train, y_train, init_params, 
                    numerical_att, categorical_att, target_att, grid_params, subset):
        self.model = model
        self.name = name
        self.folder = folder
        self.version = version
        self.desc = desc
        self.X_train = X_train
        self.y_train = y_train
        self.init_params = init_params
        self.numerical_att = numerical_att
        self.categorical_att = categorical_att
        self.target_att = target_att
        self.grid_params = grid_params
        self.subset = subset
        self.train_log_dir = './logs/'
        self.model_save_path = './models/saved_models/'

    def train(self):

        '''
        Model Fitting and Training
        Save pickle models into saved_models

        The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file
        '''

        ## start timer for runtime
        time_start = time.time()

        LOG.info('Start model training .....')

        LOG.info('..... grid searching')

        grid =  GridSearchCV(self.model, param_grid=vars(self.grid_params), n_jobs=5,
                   cv=StratifiedKFold(n_splits=5, random_state=0, shuffle=True),
                   scoring='roc_auc',
                   verbose=2, refit=True)
 
        grid.fit(self.X_train, self.y_train)

        best_params = grid.best_params_
        best_params = {re.sub("clf__","",key):value for key,value in best_params.items()}

        ## fit model on training data
        final_model = XGBClassifier(**best_params, use_label_encoder=False)
        final_model.fit(self.X_train,self.y_train)

        # save model pickel here
        save_path = os.path.join(self.model_save_path, self.name, self.folder)
        os.makedirs(save_path, exist_ok = True)

        if self.subset:
            # saving subset trained model
            LOG.info(f"saving model(subset): {self.name + '_' + self.folder + '.' + self.version + '-subset'}")
            pickle.dump(final_model, open(os.path.join(save_path, self.name + '_' + \
                self.folder + '.' + self.version + '-subset.pickle'),'wb'))
        else:
            # saving full trained model
            LOG.info(f"saving model: {self.name + '_' + self.folder + '.' + self.version}")
            pickle.dump(final_model, open(os.path.join(save_path, self.name + '_' + \
                self.folder + '.' + self.version + '.pickle'),'wb'))

            # saving full trained model
            LOG.info(f"saving latest training data for model: {self.name + '_' + self.folder + '.' + self.version}")
            data_file = os.path.join(save_path, f"{self.name + '_' + self.folder + '.' + self.version}" +'-train_data.pickle')
            with open(data_file,'wb') as tmp:
                pickle.dump({'y':self.y_train,'X':self.X_train},tmp)

        LOG.info(f"saved model: {save_path}")
        LOG.info("Model training completed")

        m, s = divmod(time.time()-time_start, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d"%(h, m, s)

        ## update the log file
        update_train_log(self.X_train.shape, runtime, self.folder + '.' + self.version, 
                            self.desc, best_params, subset=self.subset)