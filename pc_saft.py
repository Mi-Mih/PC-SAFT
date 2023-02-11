import numpy as np
import pandas as pd
from math import pi
import logging


class PCSAFT:
    def __init__(self, k=None, x=None, m=None, rho=None, sigma=None, eps=None, temperature=None):
        """
        физические параметры
        :param k: параметр бинарного взаимодействия безразмерный массив
        :param x: массив мольных долей компонентов Моль
        :param m: массив кол-ва сегментов в молекуле компонента безразмерный
        :param rho: плотность
        :param sigma: массив диаметров сегментов нм
        :param eps: массив энергетических параметров сегментов K
        :param mean_m: усреднённое число сегментов
        :param temperature: температура K
        """
        self.k = k
        self.x = np.array(x)
        self.m = np.array(m)
        self.rho = rho
        self.sigma = np.array(sigma)
        self.boltzmann = 1.380649e-23
        self.eps = np.array(eps)
        self.mean_m = np.sum(np.array(self.x) * np.array(self.m))
        self.temperature = temperature
        self.d = np.array(self.sigma) * (1 - 0.12 * np.exp(-3 * np.array(self.eps) / (1 * self.temperature)))#температурно-зависимый диаметр сегмента нм
        self.eta = (pi / 6) * self.rho * np.sum(
            np.array(self.x) * np.array(self.m) * np.array([x ** 3 for x in self.d]))#Packing fraction безразмерный

        # параметры модели
        self.z = None # сжимаемость
        self.pressure = None
        self.fugacity_coeffs = None

        # logger
        logging.basicConfig(level=10)
        self.logger = logging.getLogger()

    # метод получения коэффициентов a и b - протестировано
    def transfom_coefs(self, coefs_matrix: pd.DataFrame) -> np.array:
        # self.logger.debug(f' а_i and b_i calculation started')
        column = coefs_matrix.columns
        return (coefs_matrix[column[0]] + (self.mean_m - 1) / self.mean_m * coefs_matrix[column[1]] + (
                (self.mean_m - 1) * (self.mean_m - 2)) / self.mean_m ** 2 * coefs_matrix[column[2]]).to_numpy()

    # метод расчёта интегралов возмущения - протестировано
    def calc_integral(self, coefs: np.array):
        # self.logger.debug(f' integral calculation started')
        array_I = np.array([])
        for i in range(len(coefs)):
            array_I = np.append(array_I, coefs[i] * self.eta ** i)
        return sum(array_I)

    # метод расчёта кси(0,1,2,3) - протестировано
    def ksi(self, n: int) -> float:
        # self.logger.debug(f' ksi_{n} calculation started')
        return (pi / 6) * self.rho * np.sum(np.array(self.x) * np.array(self.m) * np.array([x ** n for x in self.d]))

    # метод комбинированияв смесях - протестировано
    def comb_sigma(self, i: int, j: int) -> float:
        # self.logger.debug(f' sigma_{i, j} calculation started')
        return 0.5 * (self.sigma[i] + self.sigma[j])

    # метод комбинирования в смесях - протестировано
    def comb_eps(self, i: int, j: int) -> float:
        # self.logger.debug(f' epsilon_{i, j} calculation started')
        return (1 - self.k[i][j]) * (self.eps[i] * self.eps[j]) ** 0.5

    # метод расчёта сжимаемости - протестировано
    def calc_c(self) -> float:
        # self.logger.debug(f' C calculation started')
        return (1 + self.mean_m * ((8 * self.eta - 2 * self.eta ** 2) / (1 - self.eta) ** 4) + (1 - self.mean_m) * (
                20 * self.eta - 27 * self.eta ** 2 + 12 * self.eta ** 3 - 2 * self.eta ** 4) / (
                        (1 - self.eta) * (2 - self.eta)) ** 2) ** (-1)

    # метод расчёта m2_eps2_sigma3 - протестировано
    def calc_m2_eps2_sigma3(self) -> float:
        # self.logger.debug(f' m^2 * eps^2 * sigma^3 calculation started')
        m2_eps2_sigma3 = 0
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                m2_eps2_sigma3 += self.x[i] * self.x[j] * self.m[i] * self.m[j] * (
                        (self.comb_eps(i, j) / (1 * self.temperature)) ** 2) * self.comb_sigma(i, j) ** 3
        return m2_eps2_sigma3

    # метод расчёта m2_eps_sigma3 - протестировано
    def calc_m2_eps_sigma3(self) -> float:
        # self.logger.debug(f' m^2 * eps * sigma^3 calculation started')
        m2_eps_sigma3 = 0
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                m2_eps_sigma3 += self.x[i] * self.x[j] * self.m[i] * self.m[j] * (
                        self.comb_eps(i, j) / (1 * self.temperature)) * self.comb_sigma(i, j) ** 3
        return m2_eps_sigma3

    # метод расчёта остаточной энергии Гельмгольца дисперсионных сил - протестировано
    def calc_alpha_disp(self):
        # self.logger.debug(f' dispersion alpha calculation started')
        a = self.transfom_coefs(pd.read_excel('a-b.xlsx'))
        b = self.transfom_coefs(pd.read_excel('a-b.xlsx'))
        I_1 = self.calc_integral(a)
        I_2 = self.calc_integral(b)

        m2_eps_sigma3 = self.calc_m2_eps_sigma3()
        m2_eps2_sigma3 = self.calc_m2_eps2_sigma3()

        return -2 * pi * self.rho * I_1 * m2_eps_sigma3 - pi * self.rho * self.calc_c() * I_2 * m2_eps2_sigma3

    # метод расчёта ост энергии Гельмгольца твёрдых сфер - протестировано
    def calc_alpha_hs(self):
        # self.logger.debug(f' alpha_hs calculation started')
        ksi0, ksi1 = self.ksi(0), self.ksi(1)
        ksi2, ksi3 = self.ksi(2), self.ksi(3)
        return (1 / self.ksi(0)) * (
                (3 * ksi1 * ksi2) / (1 - ksi3) + ksi2 ** 3 / (ksi3 * (1 - ksi3) ** 2) + np.log(1 - ksi3) * (
                (ksi2 ** 3 / ksi3 ** 3) - ksi0))

    # метод радиальная функция распределения в системе твёрдых сфер - протестировано
    def radial_func_distr(self):
        # self.logger.debug(f' radial function distribution calculation started')
        ksi2, ksi3 = self.ksi(2), self.ksi(3)
        g = np.zeros((len(self.x), len(self.x)))
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                g[i][j] = 1 / (1 - ksi3) + (self.d[i] * self.d[j] / (self.d[i] + self.d[j])) * 3 * ksi2 / (
                        1 - ksi3) ** 2 + (self.d[i] * self.d[j] / (self.d[i] + self.d[j])) ** 2 * 3 * ksi2 ** 2 / (
                                  1 - ksi3) ** 3
        return g

    # метод расчёта ост энергии Гельмгольца твёрдых цепей - протестировано
    def calc_alpha_chain(self):
        # self.logger.debug(f' alpha hard chain calculation started')
        alpha_hard_sphere = self.calc_alpha_hs()
        g = self.radial_func_distr()
        return self.mean_m * alpha_hard_sphere - np.sum(self.x * (np.array(self.m) - 1) * np.log(g.diagonal()))

    # метод расчёта остаточной энергии Гельмгольца - протестировано
    def calc_energy_helmholtz(self):
        # self.logger.debug(f' energy of helmholtz calculation started')
        return self.calc_alpha_chain() + self.calc_alpha_disp()

    # метод расчёта коэффициента сжимаемости - протестирован
    def calc_z(self):
        # self.logger.debug(f' z calculation started')
        rho_1 = self.rho
        eta_array = np.array([])
        for i in [-2, -1, 1, 2]:
            eta_array = np.append(eta_array, self.eta + i * self.eta * 0.0001)
        rho_array = np.array([])
        for eta in eta_array:
            rho_array = np.append(rho_array, 6 * eta / (np.sum(np.array(self.x) * np.array(self.m) * self.d ** 3) * pi))
        alpha_res_array = np.array([])
        for rho in rho_array:
            self.rho = rho
            alpha_res_array = np.append(alpha_res_array, self.calc_energy_helmholtz())
        self.rho = rho_1
        return 1 + self.eta * (
                alpha_res_array[0] - 8 * alpha_res_array[1] + 8 * alpha_res_array[2] - alpha_res_array[3]) / (
                       12 * self.eta * 0.0001)

    # метод расчёта давления - протестирован
    def calc_pressure(self):
        # self.logger.debug(f' pressure calculation started')
        z = self.calc_z()
        self.pressure = z * self.boltzmann * self.temperature * self.rho * 10e24

    # метод расчёта частных производных - протестирован
    def calc_partial_x(self):
        # self.logger.debug(f' partials x calculation started')
        x_1 = self.x
        partial_array = np.array([])
        for i in range(len(self.x)):
            alpha_res_array = np.array([])
            h = 12 * self.x[i] * 0.0001

            for j in [-2, -1, 1, 2]:
                self.x[i] += j * 0.0001 * self.x[i]
                alpha_res_array = np.append(alpha_res_array, self.calc_energy_helmholtz())
            partial_array = np.append(partial_array,
                                      (alpha_res_array[0] - 8 * alpha_res_array[1] + 8 * alpha_res_array[2]
                                       - alpha_res_array[3]) / h)
        self.x = x_1
        return partial_array

    # метод расчёта коэффициентов летучести - протестирован
    def calc_fugacity_coeff(self):
        # self.logger.debug(f' fugacity coeffs calculation started')
        alpha_res = self.calc_energy_helmholtz()
        z = self.calc_z()
        partials = self.calc_partial_x()
        fugacity_coeffs = np.array([])
        for i in range(len(self.x)):
            fugacity_coeffs = np.append(fugacity_coeffs, np.exp(
                alpha_res + (z - 1) + partials[i] - np.sum(self.x * partials) - np.log(z)))
        self.fugacity_coeffs = fugacity_coeffs

    # метод расчёта мольных долей - протестирован
    # current - текущая фаза, forecast - неизвестная фаза
    # меняем плотность, молярный объём и eta
    def calc_lv_proportion(self, proportion, fugacity_current, fugacity_forecast):
        # self.logger.debug(f' vapour-liquid calculation started')
        return (proportion * fugacity_current) / fugacity_forecast


# метод расчёта парожидкостного равновесия
def launch_pcsaft(state_1: dict, state_2: dict, current_state='liquid'):
    dicts_correct = {'cur_eta': [], 'forec_eta': [], 'error': [], 'cur_x': [], 'forec_x': []}
    if current_state == 'liquid':
        cur_array = [0.0003, 0.0005, 0.0008]
        forec_array = [0.05, 0.03, 0.02, 0.01]
    else:
        cur_array = [0.05, 0.03, 0.02, 0.01]
        forec_array = [0.0002, 0.0003, 0.0005, 0.0008, 0.001]
    current = PCSAFT(**state_1)
    forecast = PCSAFT(**state_2)

    for eta_cur in cur_array:  # доделать
        current.eta = eta_cur
        current.rho = 6 * eta_cur / np.sum(
            np.array(forecast.x) * np.array(forecast.m) * np.array([x ** 3 for x in current.d]))
        current.calc_pressure()
        current.calc_fugacity_coeff()

        for eta_forec in forec_array:  # доделать
            forecast.eta = eta_forec
            forecast.rho = 6 * eta_forec / np.sum(
                np.array(current.x) * np.array(current.m) * np.array([x ** 3 for x in forecast.d]))
            forecast.calc_pressure()
            forecast.calc_fugacity_coeff()

            forecast.x = forecast.calc_lv_proportion(current.x, current.fugacity_coeffs, forecast.fugacity_coeffs)
            if len(forecast.x) != len([y for y in forecast.x if y == y]):
                continue
            x_scaled = [mol / sum(forecast.x) for mol in forecast.x]
            forecast.x = x_scaled

            dicts_correct['cur_eta'].append(eta_cur)
            dicts_correct['forec_eta'].append(eta_forec)
            dicts_correct['error'].append(abs(current.pressure - forecast.pressure))
            dicts_correct['cur_x'].append(current.x)
            dicts_correct['forec_x'].append(x_scaled)

    df_correct = pd.DataFrame(dicts_correct)
    return df_correct.iloc[df_correct.error.idxmin()]


if __name__ == '__main__':
    pass
