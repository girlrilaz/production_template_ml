import os
import sys
import unittest

class MakeDataTest(unittest.TestCase):
    """
    test the final data creation functionality
    """
        
    def test_01_folder(self):
        """
        ensure the raw data folder (data/raw) is not empty
        """
        input_folder = os.path.join('.','data', 'raw')
        if [f for f in os.listdir(input_folder) if not f.startswith('.')] == []:
            raise ValueError("Folder 'data/raw' is empty. Please ensure there is atleast one data file in this folder.")
        # else: 
        #     print("Raw folder 'not-empty' test passed")

### Run the tests
if __name__ == '__main__':
    unittest.main()

