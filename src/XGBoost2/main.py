# -*- coding: utf-8 -*-
""" main.py """

# standard libraries
import sys

# internal
from configs.module.config import CFG
from models.XGboost1 import XGB


def run():
    """Builds model, loads data, trains and evaluates"""

    if sys.argv[1] == "subset": 
        subset = True
    else:
        subset = False

    model = XGB(CFG)
    model.load_data(subset=subset)
    model.build()
    model.train(subset=subset)
    model.evaluate(subset=subset)


if __name__ == '__main__':
    run()