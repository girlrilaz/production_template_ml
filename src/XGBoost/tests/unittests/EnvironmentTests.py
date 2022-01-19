"""Environment unit test"""
import sys
import unittest

REQUIRED_PYTHON = "python3"

class EnvironmentTest(unittest.TestCase):
    """
    test the essential functionality
    """
    def test_01_environment(self):
        """
        ensure environment requirement is satisfied
        """

        system_major = sys.version_info.major
        if REQUIRED_PYTHON == "python":
            required_major = 2
        elif REQUIRED_PYTHON == "python3":
            required_major = 3
        else:
            raise ValueError(f"Unrecognized python interpreter: {REQUIRED_PYTHON}")

        if system_major != required_major:
            raise TypeError(
                f"This project requires Python {required_major}. Found: Python {sys.version}")
        else:
            print(">>> Development environment passes all tests!")


### Run the tests
if __name__ == '__main__':
    unittest.main()
