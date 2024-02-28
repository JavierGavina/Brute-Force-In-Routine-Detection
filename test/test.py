import unittest
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import main
from unittest.mock import patch, MagicMock


class TestMain(unittest.TestCase):

    def test_process_sequence(self):
        sequence = ["1", "2", "", "4"]
        expected_output = np.array([1, 2, np.nan, 4])
        np.testing.assert_array_equal(main.process_sequence(sequence), expected_output)

    @patch('argparse.ArgumentParser.parse_args')
    def test_load_data(self, mock_args):
        # Create a test csv file
        with open('test.csv', 'w') as file:
            file.write("Year,Month,Day,Sequence\n")
            file.write("2022,12,1,1,2,,4\n")

        expected_output = pd.DataFrame({
            "Year": [2022],
            "Month": [12],
            "Day": [1],
            "Sequence": [np.array([1, 2, np.nan, 4])]
        })

        # Mock the argparse.ArgumentParser().parse_args() method
        mock_args.return_value = MagicMock(data_dir='test.csv')

        pd.testing.assert_frame_equal(main.load_data('test.csv'), expected_output)
        # drop the test file
        os.remove('test.csv')


if __name__ == '__main__':
    unittest.main()
