# -*- coding: utf-8 -*-
"""Config class"""

# standard library
import json
import yaml

class Config:
    """Config class which contains data, train and model hyperparameters"""
    def __init__(self, data, train, model):
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from configs/module"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)

    @classmethod
    def from_json_file(cls):
        """Creates config from configs/json"""
        with open('configs/json/config.json', encoding="utf-8") as filename:
            params = json.load(filename, object_hook=HelperObject)
        return cls(params.data, params.train, params.model)

    @classmethod
    def from_yaml(cls):
        """Creates config from configs/yaml"""
        with open('configs/yaml/config.yml', encoding="utf-8") as filename:
            params = json.loads(json.dumps(yaml.safe_load(filename)), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)

class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)