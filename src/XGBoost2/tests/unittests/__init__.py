"""initialize unittests"""

import unittest
import getopt
import sys
import os

from .EnvironmentTests import *
from .MakeDataTests import *
from .ApiTests import *
from .ModelTests import *
from .LoggerTests import *

## parse inputs
try:
    optlist, args = getopt.getopt(sys.argv[1:],'v')
except getopt.GetoptError:
    print(getopt.GetoptError)
    print(sys.argv[0] + "-v")
    print("... the verbose flag (-v) may be used")
    sys.exit()

VERBOSE = False
RUNALL = False

sys.path.append(os.path.realpath(os.path.dirname(__file__)))

for o, a in optlist:
    if o == '-v':
        VERBOSE = True

## Environment tests
EnvironmentTestSuite = unittest.TestLoader().loadTestsFromTestCase(EnvironmentTest)

## Made data tests
MakeDataTestSuite = unittest.TestLoader().loadTestsFromTestCase(MakeDataTest)

## api tests
ApiTestSuite = unittest.TestLoader().loadTestsFromTestCase(ApiTest)

## model tests
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(ModelTest)

## logger tests
LoggerTestSuite = unittest.TestLoader().loadTestsFromTestCase(LoggerTest)

MainSuite = unittest.TestSuite([EnvironmentTestSuite, MakeDataTestSuite, ModelTestSuite, LoggerTestSuite, ApiTestSuite])
