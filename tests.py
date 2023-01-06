import unittest
from pc_saft import PCSAFT
import numpy as np
import pandas as pd
from pc_saft import launch_pcsaft


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.pc_saft = PCSAFT(x=[0.1000, 0.3000, 0.6000],
                              m=[1, 1.6069, 2.002],
                              sigma=[3.7039, 3.5206, 3.6184],
                              temperature=233.15,
                              boltzmann=1.380649e-23,
                              rho=9.51e-03,
                              eps=[150.03, 191.42, 208.11],
                              k=[0, 3e-4, 1.15e-2],
                              )

    def test_calc_mean_m(self):
        self.assertEqual(1.783270, self.pc_saft.mean_m,
                         f'Разница составляет {1.783270 - self.pc_saft.mean_m}')

    def test_calc_d(self):
        true_d = [3.6394, 3.4846, 3.5886]
        self.assertCountEqual(true_d, self.pc_saft.d,
                              f'Разница составляет {true_d - self.pc_saft.d}')

    def test_calc_ksi(self):
        # self.pc_saft.calc_d()
        ksi_0 = self.pc_saft.ksi(n=0)
        ksi_1 = self.pc_saft.ksi(n=1)
        ksi_2 = self.pc_saft.ksi(n=2)
        ksi_3 = self.pc_saft.ksi(n=3)

        self.assertEqual(8.88e-03, ksi_0,
                         f'Разница составляет {8.88e-03 - ksi_0}')
        self.assertEqual(round(3.16e-02, 4), round(ksi_1, 4),
                         f'Разница составляет {3.16e-02 - ksi_1}')
        self.assertEqual(1.13e-01, ksi_2,
                         f'Разница составляет {1.13e-01 - ksi_2}')
        self.assertEqual(4.02e-01, ksi_3,
                         f'Разница составляет {4.02e-01 - ksi_3}')

    def test_comb_sigma(self):
        matrix = np.array([[3.704, 3.612, 3.661],
                           [3.612, 3.521, 3.570],
                           [3.661, 3.570, 3.618]])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.assertEqual(matrix[i][j], self.pc_saft.comb_sigma(i, j),
                                 f'Разница составляет {matrix[i][j] - self.pc_saft.comb_sigma(i, j)}')

    def test_comb_eps(self):
        matrix = np.array([[1.50E+02, 1.69E+02, 1.75E+02],
                           [1.69E+02, 1.91E+02, 1.99E+02],
                           [1.75E+02, 1.99E+02, 2.08E+02]])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.assertEqual(matrix[i][j], self.pc_saft.comb_eps(i, j),
                                 f'Разница составляет {round(matrix[i][j], 4) - self.pc_saft.comb_eps(i, j)}')

    def test_calc_c(self):
        # self.pc_saft.calc_d()
        # self.pc_saft.calc_m()
        self.pc_saft.eta = self.pc_saft.ksi(3)
        self.assertEqual(2.662e-02, self.pc_saft.calc_c(),
                         f'Разница составляет {2.662e-02 - self.pc_saft.calc_c()}')

    def test_calc_alpha_hs(self):
        # self.pc_saft.calc_d()
        self.assertEqual(3.139757, self.pc_saft.calc_alpha_hs(),
                         f'Разница составляет {3.139757 - self.pc_saft.calc_alpha_hs()}')

    def test_radial_func_distr(self):
        # self.pc_saft.calc_m()
        # self.pc_saft.calc_d()
        true_g = [3.7877, 3.6817, 3.7527]
        g = self.pc_saft.radial_func_distr()
        self.assertCountEqual(true_g, np.diagonal(g),
                              f'Разница составляет {true_g - np.diagonal(g)}')

    def test_transfom_coefs(self):
        # self.pc_saft.calc_m()
        true_a = [7.80e-01, 6.94e-01, 1.55e+00, -1.70e+01, 6.93e+01, -1.24e+02, 7.69e+01]
        true_b = [4.66e-01, 2.56e+00, -1.80e+00, -2.97e+01, 1.14e+02, 1.30e+02, -4.27e+02]
        a = self.pc_saft.transfom_coefs(pd.read_excel('a-b.xlsx', sheet_name='a'))
        b = self.pc_saft.transfom_coefs(pd.read_excel('a-b.xlsx', sheet_name='b', ))
        self.assertCountEqual(true_a, a,
                              f'Разница составляет {true_a - a}')
        self.assertCountEqual(true_b, b,
                              f'Разница составляет {true_b - b}')

    def test_calc_integral(self):
        # self.pc_saft.calc_d()
        # self.pc_saft.calc_m()
        self.pc_saft.eta = self.pc_saft.ksi(3)
        a = self.pc_saft.transfom_coefs(pd.read_excel('a-b.xlsx', sheet_name='a'))
        b = self.pc_saft.transfom_coefs(pd.read_excel('a-b.xlsx', sheet_name='b'))

        Ia = self.pc_saft.calc_integral(a)
        Ib = self.pc_saft.calc_integral(b)

        self.assertEqual(1.038612, Ia,
                         f'Разница составляет {1.038612 - Ia}')
        self.assertEqual(1.811019, Ib,
                         f'Разница составляет {1.811019 - Ib}')

    def test_calc_m2_eps_sigma3(self):
        m2es3 = self.pc_saft.calc_m2_eps_sigma3()
        self.assertEqual(1.267e+02, m2es3,
                         f'Разница составляет {1.267e+02 - m2es3}')

    def test_calc_m2_eps2_sigma3(self):
        m2e2s3 = self.pc_saft.calc_m2_eps2_sigma3()
        self.assertEqual(1.087e+02, m2e2s3,
                         f'Разница составляет {1.087e+02 - m2e2s3}')

    def test_calc_alpha_disp(self):
        # self.pc_saft.calc_d()
        # self.pc_saft.calc_m()
        alpha_disp = self.pc_saft.calc_alpha_disp()
        self.assertEqual(-8.1404, alpha_disp,
                         f'Разница составляет {-8.1404 - alpha_disp}')

    def test_calc_alpha_chain(self):
        # self.pc_saft.calc_m()
        # self.pc_saft.calc_d()
        alpha_chain = self.pc_saft.calc_alpha_chain()
        self.assertEqual(4.566659053, alpha_chain,
                         f'Разница составляет {4.566659053 - alpha_chain}')

    def test_calc_energy_helmholtz(self):
        alpha_res = self.pc_saft.calc_energy_helmholtz()
        self.assertEqual(-3.57371, alpha_res,
                         f'Разница составляет {-3.57371 - alpha_res}')

    def test_calc_z(self):
        # self.pc_saft.calc_m()
        # self.pc_saft.calc_d()
        self.pc_saft.eta = self.pc_saft.ksi(3)

        self.assertEqual(1.64049719, self.pc_saft.calc_z(),
                         f'Разница составляет {1.64049719 - self.pc_saft.calc_z()}')

    def test_calc_pressure(self):
        # проверка на работоспособность метода, правильного зн-я нет
        self.pc_saft.calc_pressure()
        self.assertEqual(0, self.pc_saft.pressure,
                         f'Разница составляет {0 - self.pc_saft.pressure}')

    def test_partial_x(self):
        true_partials = np.array([0.4887, -1.786, -3.244])
        partials = self.pc_saft.calc_partial_x()
        self.assertCountEqual(true_partials, partials,
                              f'Разница составляет {true_partials - partials}')

    def test_fugacity_coeffs(self):
        self.pc_saft.calc_fugacity_coeff()
        true_fugacity_coeffs = np.array([0.603, 0.06199, 0.01442])
        self.assertCountEqual(true_fugacity_coeffs, self.pc_saft.fugacity_coeffs,
                              f'Разница составляет {true_fugacity_coeffs - self.pc_saft.fugacity_coeffs}')

    def test_launch_pcsaft(self):
        liquid_state = {'x': [0.1000, 0.3000, 0.6000],
                        'm': [1, 1.6069, 2.002],
                        'sigma': [3.7039, 3.5206, 3.6184],
                        'temperature': 233.15,
                        'boltzmann': 1.380649e-23,
                        'rho': 1.18e-2,
                        'eps': [150.03, 191.42, 208.11],
                        'k': [0, 3e-4, 1.15e-2]}

        vapour_state = {'x': [0.1000, 0.3000, 0.6000],
                        'm': [1, 1.6069, 2.002],
                        'sigma': [3.7039, 3.5206, 3.6184],
                        'temperature': 233.15,
                        'boltzmann': 1.380649e-23,
                        'rho': 2.37e-12,
                        'eps': [150.03, 191.42, 208.11],
                        'k': [0, 3e-4, 1.15e-2]}

        df = launch_pcsaft(liquid_state, vapour_state)
        print(df)


if __name__ == '__main__':
    unittest.main()
