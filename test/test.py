import unittest
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import main
from main import DRFL
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


class TestDRFL(unittest.TestCase):

    def setUp(self):
        self.drfl = DRFL(m=4, R=10, C=4, G=60, epsilon=0.5)
        self.time_series = pd.Series(np.random.randint(0, 100, size=(100,)))
        self.time_series.index = pd.date_range(start='1/1/2022', periods=100)

    def test_Mag(self):
        S = np.array([1, 2, 3, 4, 5])
        self.assertEqual(self.drfl._DRFL__Mag(S), 5)

    def test_Dist(self):
        S1 = np.array([1, 2, 3, 4, 5])
        S2 = np.array([6, 7, 8, 9, 10])
        self.assertEqual(self.drfl._DRFL__Dist(S1, S2), 5)

    def test_NTM(self):
        Si = np.array([1, 2, 3, 4, 5])
        Sj = np.array([6, 7, 8, 9, 10])
        R = 10
        self.assertTrue(self.drfl._DRFL__NTM(Si, Sj, R))

    def test_IsOverlap(self):
        Sm_i = np.array([1, 2, 3, 4, 5])
        Sn_j = np.array([6, 7, 8, 9, 10])
        i = 1
        j = 3
        self.assertFalse(self.drfl._DRFL__IsOverlap(Sm_i, Sn_j, i, j))

    def test_OLTest(self):
        Sm = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10]), np.array([11, 12, 13, 14, 15])]
        Sn = [np.array([16, 17, 18, 19, 20]), np.array([21, 22, 23, 24, 25]), np.array([26, 27, 28, 29, 30])]
        epsilon = 0.5
        Km, Kn = self.drfl._DRFL__OLTest(Sm, Sn, epsilon)
        self.assertTrue(Km)
        self.assertTrue(Kn)

    def test_SubGroup(self):
        S = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])]
        R = 10
        C = 2
        G = 10
        dates = [pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-02')]
        B = self.drfl._DRFL__SubGroup(S, R, C, G, dates)
        self.assertEqual(len(B), 2)

    def test_extract_subsequences(self):
        time_series = pd.Series(np.random.randint(0, 100, size=(100,)))
        subsequences, dates = self.drfl._DRFL__extract_subsequences(time_series, self.drfl.m)
        self.assertEqual(len(subsequences), len(time_series) - self.drfl.m + 1)
        self.assertEqual(len(dates), len(time_series) - self.drfl.m + 1)

    def test_decide_Km_Kn(self):
        len_Sm = 5
        len_Sn = 6
        Mag_Sm = 10
        Mag_Sn = 20
        N = 3
        epsilon = 0.5
        Km, Kn = self.drfl._DRFL__decide_Km_Kn(len_Sm, len_Sn, Mag_Sm, Mag_Sn, N, epsilon)
        self.assertFalse(Km)
        self.assertTrue(Kn)


if __name__ == '__main__':
    unittest.main()
