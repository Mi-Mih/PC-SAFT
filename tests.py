import unittest
from pc_saft import PCSAFT
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.pc_saft = PCSAFT(x=[0.1000, 0.3000, 0.6000],
                              m=[1, 1.6069, 2.002],
                              sigma=[3.7039, 3.5206, 3.6184],
                              temperature=233.15,
                              boltzmann=1.380649e-23,
                              rho=9.51e-03,
                              eps=[150.03, 191.42, 208.11],
                              k=[0.01,0.01,0.01])

    def test_calc_mean_m(self):
        self.pc_saft.calc_m()
        self.assertEqual(round(1.783270, 4), round(self.pc_saft.mean_m, 4),
                         f'Разница составляет {round(1.783270, 4) - round(self.pc_saft.mean_m, 4)}')

    def test_calc_d(self):
        self.pc_saft.calc_d()
        self.assertCountEqual([3.7039, 3.5206, 3.6184], self.pc_saft.d)

    def test_calc_ksi(self):
        self.pc_saft.calc_d()
        ksi_0 = self.pc_saft.ksi(n=0)
        ksi_1 = self.pc_saft.ksi(n=1)
        ksi_2 = self.pc_saft.ksi(n=2)
        ksi_3 = self.pc_saft.ksi(n=3)

        self.assertEqual(round(8.88e-03, 4), round(ksi_0, 4),
                         f'Разница составляет {round(8.88e-03, 4) - round(ksi_0, 4)}')
        self.assertEqual(round(3.16e-02, 4), round(ksi_1, 4),
                         f'Разница составляет {round(3.16e-02, 4) - round(ksi_1, 4)}')
        self.assertEqual(round(1.13e-01, 4), round(ksi_2, 4),
                         f'Разница составляет {round(1.13e-01, 4) - round(ksi_2, 4)}')
        self.assertEqual(round(4.02e-01, 4), round(ksi_3, 4),
                         f'Разница составляет {round(4.02e-01, 4) - round(ksi_3, 4)}')

    def test_comb_sigma(self):
        matrix = np.array([[3.704, 3.612, 3.661],
                           [3.612, 3.521, 3.570],
                           [3.661, 3.570, 3.618]])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.assertEqual(round(matrix[i][j], 4), round(self.pc_saft.comb_sigma(i,j), 4),
                                 f'Разница составляет {round(matrix[i][j], 4) - round(self.pc_saft.comb_sigma(i,j), 4)}')

    def test_comb_eps(self):
        matrix = np.array([[1.50E+02,1.69E+02,1.75E+02],
                          [1.69E+02,	1.91E+02,1.99E+02],
                          [1.75E+02,	1.99E+02,	2.08E+02]])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.assertEqual(round(matrix[i][j], 4), round(self.pc_saft.comb_eps(i,j), 4),
                                 f'Разница составляет {round(matrix[i][j], 4) - round(self.pc_saft.comb_eps(i,j), 4)}')

if __name__ == '__main__':
    unittest.main()
