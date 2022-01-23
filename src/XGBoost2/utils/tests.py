#!/usr/bin/python
"""Run all unit testing"""
#standard library
import unittest
import sys
sys.path.append('./tests')

#internal
from tests.unittests import *

unittest.main()
