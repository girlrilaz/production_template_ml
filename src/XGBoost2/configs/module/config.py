# -*- coding: utf-8 -*-
# """Model config in json format"""

CFG = {
    "data": {
        "source" : "local",
        "type" : "csv",
        "path": "data/raw/bank.csv",
        "bucket": "",
        "subset_n_frac" : 0.5,
        "numerical_att": ["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
        "categorical_att" : ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
    },
    "train": {
        "target_att": ["y"],
        "test_size": 0.33,
        "random_state": 123,
        "metrics": ["accuracy"]
    },
    "model": {
        "name" : "XGBoost",
        "folder": "1",
        "version": "0.0",
        "short_description": "XGBoost on imbalanced bank data",
        "params": {
            "nthread": [4],
            "n_estimators": [20],
            "objective": ["binary:logistic"], 
            "learning_rate" : [0.5],
            "eval_metric": ["error"], 
            "eta": [0.3], 
            "gamma": [0], 
            "max_depth": [7], 
            "min_child_weight": [5], 
            "max_delta_step": [0], 
            "subsample": [1], 
            "seed": [0], 
            "scale_pos_weight": [1]},
        "grid_params": {
            "nthread":[4],
            "n_estimators": [20, 40],
            "objective": ["binary:logistic"],
            "learning_rate" : [0.25, 0.5],
            "eval_metric": ["error"],
            "eta": [0.3],
            "gamma": [0],
            "max_depth": [6],
            "min_child_weight": [4],
            "max_delta_step": [0],
            "subsample": [1],
            "seed": [0],
            "scale_pos_weight": [1]
        }
    }
}


### Model Parameter Explanations

       # select the type of model to run at each iternation we have the options of tree and linear models
        ##param['booster'] = 'gbtree'
        
        # since we want the output to have the probability also, we will use the logistic objective.
        ##param['objective'] = 'binary:logistic'
        
        # lets use the error as the eval metrics i.e in each boosting steps we will reduce error
        ##param["eval_metric"] = "error"
        
        # eta is like learning rate and it makes the model more robust by shrinking the weights at each iter
        ##param['eta'] = 0.3
        
        # gamma controls the minimum loss reduction to split and it should be tuned.
        ##param['gamma'] = 0

        # maximum depth of a tree to control the over fitting. should be tuned with cv
        ##param['max_depth'] = 10
        
        # minimum number of samples for the leaf and is used to control overfitting. We will use lower values, as we have
        # class imbalance and if we set high then accuracy of minory class will be affected
        ##param['min_child_weight']=1
        
        # maximum delta step from previous iteration for each tree. Higher the value (i.e non zero), more conservative we are
        ##param['max_delta_step'] = 0
        
        # as explained before in boosting each tree is build using samples from prev iteration with replace and this specify 
        # the fraction of data to be used for each tree. typically values slightly less than 1 makes it robust.
        ##param['subsample']= 1
        
        # as explained before boosting use subset of rows and also subset of columns. this controls the subset of cols as fraction
        ##param['colsample_bytree']=1
        
        # control the verbosity
        ##param['silent'] = 0
        
        # random seed for reproducibility
        ##param['seed'] = 0
        
        # set the initial prediction score i.e global bias 
        ##param['base_score'] = 0.5
        
        # how much weight to give the positive sample, in future we will change it but for now lets put it 1
        ##param['scale_pos_weight']= 1