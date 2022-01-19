# -*- coding: utf-8 -*-
"""Model config in json format"""

from pandera import Column, DataFrameSchema, Check, Index

SCHEMA = DataFrameSchema (
  {
    "age": Column(int, required=True),
    "job": Column(str, required=True),
    "marital": Column(str, required=True),
    "education": Column(str, required=True),
    "default": Column(str, required=True),
    "balance": Column(int, required=True),
    "housing": Column(str, required=True),
    "loan": Column(str, required=True),
    "contact": Column(str, required=True),
    "day": Column(int, required=True),
    "month": Column(str, required=True),
    "duration": Column(int, required=True),
    "campaign": Column(int, required=True),
    "pdays": Column(int, required=True),
    "previous": Column(int, required=True),
    "poutcome": Column(str, required=True),
    "y": Column(str, required=True),
    },
    coerce=False,  # True : allows null values in columns
    strict=False, # False : allows dataset to have extra columns 
)