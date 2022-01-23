# -*- coding: utf-8 -*-
"""Model Evaluator"""

# standard library
import os
import time
import pickle
import pandas as pd

# external
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

# internal
# from utils.visualize_WIP import display
from utils.config import Config
from configs.module.config import CFG
from utils.dataloader import DataLoader
from utils.logger import get_logger
from utils.visualize import plot_cm, plot_pr_roc, plot_pr_vs_th
from utils.logger import update_evaluation_log

LOG = get_logger('xgboost_evaluator')

class XGBEvaluator:
    def __init__(self, test_dataset, dataset, X_test, y_test, subset):
        self.config = Config.from_json(CFG)
        self.predictions = {}
        self.test_dataset = test_dataset
        self.dataset = dataset
        self.X_test = X_test
        self.y_test = y_test
        self.subset = subset
        self.model_name = self.config.model.name
        self.model_folder = self.config.model.folder
        self.model_version = self.config.model.version
        self.predictions = {}
        self.report_path = './evaluation/report'
        self.plots_path = './evaluation/plots'

    def test_data_validation(self):

        ## schema checks
        try:
            validate = DataLoader().validate_schema(self.test_dataset)
            if validate is None:
                LOG.info("PASS: Test data validation passed.")
        except:
            raise Exception("ERROR - FAIL:(model_evaluation) - invalid input schema.")

        ## input checks
        if isinstance(self.test_dataset,dict):
            self.test_dataset = pd.DataFrame(self.test_dataset)
        elif isinstance(self.test_dataset,pd.DataFrame):
            pass
        else:
            raise Exception(f"ERROR - FAIL:(model_evaluation) - invalid input. {self.test_dataset} was given")

        ## features check
        test_features = sorted(self.test_dataset.columns.drop(['y']).tolist())
        data_features = sorted(self.dataset.columns.drop(['y']).tolist())
        if test_features != data_features:
            print(f"test features: {','.join(test_features)}")
            raise Exception("ERROR - FAIL:(model_evaluation) - invalid features present")

    def model_load(self):

        if self.subset:
            model_pickle_name = self.model_name + '_' + self.model_folder + '.' + self.model_version + '-subset.pickle'
            saved_model = os.path.join('models', 'saved_models', self.model_name, self.model_folder, model_pickle_name)
            LOG.info(f"..... loading trained subset model {saved_model}")
        else:
            model_pickle_name = self.model_name + '_' + self.model_folder + '.' + self.model_version + '.pickle'
            saved_model = os.path.join('models', 'saved_models', self.model_name, self.model_folder, model_pickle_name)
            LOG.info(f"..... loading model {saved_model}")

        if not os.path.exists(saved_model):
            exc = (f"model '{saved_model}' cannot be found. Did you train the full model?")
            raise Exception(exc)

        return pickle.load(open(saved_model, 'rb'))

    def model_predict(self, model):

        ## make prediction and gather data for log entry
        LOG.info("..... starting model prediction on test data")

        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        self.predictions = [round(value) for value in y_pred]

        LOG.info('Model evaluation completed')

        self.predictions = {'y_pred':y_pred,'y_proba':y_proba}

        return self.predictions

    def evaluation_report(self, model):

        ## start timer for runtime
        time_start = time.time()

        model_name_version = self.model_name + '_' + self.model_folder + '.' + self.model_version
        os.makedirs(self.report_path, exist_ok = True)

        print("\n Accuracy Report - ", model_name_version, "")

        ## GENERATE METRIC REPORTS
        # evaluate predictions using train_test split - quicker
        accuracy = accuracy_score(self.y_test, self.predictions['y_pred'])
        simple_acc = round(accuracy * 100.0,2)
        acc_1 = f"\n Train-Test split accuracy: {simple_acc} %"
        print(acc_1)

        # evaluate predictions using Kfold method - good for sets model has not seen
        kfold = KFold(n_splits=10, random_state=7, shuffle = True)
        results = cross_val_score(model, self.X_test, self.y_test, cv=kfold)
        k_fold_mean = round(results.mean()*100,2)
        k_fold_std = round(results.std()*100,2)
        acc_2 = f" K-fold validation accuracy (std): {k_fold_mean} % ({k_fold_std} %)"
        print(acc_2)

        # evaluate predictions using Stratified Kfold method - good for multiple classes or imbalanced dataset
        stratified_kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle = True)
        s_results = cross_val_score(model, self.X_test, self.y_test, cv=stratified_kfold)
        s_k_fold_mean = round(s_results.mean()*100,2)
        s_k_fold_std = round(s_results.std()*100,2)
        acc_3 = f" Stratified K-fold validation accuracy (std): {s_k_fold_mean} % ({s_k_fold_std} %)"
        print(acc_3)

        # evaluate predictions using confusion matrix
        print("\n Classification Report - ", model_name_version, "\n\n")
        print(classification_report(self.y_test, self.predictions['y_pred'])) #output_dict=True

        # evaluate predictions using confusion matrix roc_auc_score
        roc_score = roc_auc_score(self.y_test, self.predictions['y_proba'][:, 1])
        roc_auc = round(roc_score,2)
        print(f"\n Receiver Operating Characteristics (ROC) Score : {roc_auc} \n\n")

        #summarize metrics in a dict
        eval_metrics = self.metrics_summary(simple_acc, k_fold_mean, k_fold_std, s_k_fold_mean, s_k_fold_std, roc_auc)

        # ## GENERATE PLOTS
        os.makedirs(self.plots_path, exist_ok = True)
        plot_pr_roc(self.y_test, self.predictions['y_proba'][:,1], self.plots_path, "", "darkorange", False, model_name_version)
        plot_cm(self.y_test, self.predictions['y_pred'], self.plots_path, model_name_version + " - Confusion Matrix")

        LOG.info("Model evaluation completed")
        LOG.info(f"Evaluation reports saved in : {self.report_path}")
        LOG.info(f"Evaluation plots saved in : {self.plots_path}")

        m, s = divmod(time.time()-time_start, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d"%(h, m, s)

        ## update the log file
        update_evaluation_log(eval_metrics, runtime, self.model_folder + '.' + self.model_version)

        return simple_acc, k_fold_mean, k_fold_std, s_k_fold_mean, s_k_fold_std, roc_auc

    def metrics_summary(self, simple_acc, k_fold_mean, k_fold_std, s_k_fold_mean, s_k_fold_std, roc_auc):

        eval_metrics = {"simple_acc": simple_acc, 
                        "k_fold_acc" : {"k_fold_mean" : k_fold_mean, "k_fold_std" : k_fold_std},
                        "s_k_fold_acc" : {"s_k_fold_mean" : s_k_fold_mean, "s_k_fold_std": s_k_fold_std},
                        "roc_auc": roc_auc
                        }
        # print(json.dumps(eval_metrics, indent = 4))
        return eval_metrics
