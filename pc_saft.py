import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from math import e


class PCSAFT:
    def __init__(self, k=None, x=None, m=None, rho=None, sigma=None, eps=None, boltzmann=None, temperature=None, h=0.001):
        """
        :param k: параметр бинарного взаимодействия
        :param x: массив мольных долей компонентов
        :param m: массив кол-ва сегментов в молекуле компонента
        :param rho: плотность
        :param sigma: массив диаметров сегментов
        :param eps: массив энергетических параметров сегментов
        :param boltzmann: постоянная Больцмана
        :param mean_m: усреднённое число сегментов
        :param eta: Packing fraction
        :param temperature: температура
        :param d: температурно-зависимый диаметр сегмента
        :param h: шаг для вычисления частной производной
        :param h: сжимаемость
        """
        self.k = k
        self.x = x
        self.m = m
        self.rho = rho
        self.sigma = sigma
        self.eps = eps
        self.boltzmann = boltzmann
        self.mean_m = None
        self.eta = None
        self.temperature = temperature
        self.d = None
        self.h = h
        self.z = None

    # метод расчёта усреднённого числа сегментов - протестировано
    def calc_m(self) -> float:
        self.mean_m = np.sum(np.array(self.x) * np.array(self.m))

    # метод получения коэффициентов a и b - протестировано
    def transfom_coefs(self, coefs_matrix: pd.DataFrame) -> np.array:
        column = coefs_matrix.columns
        return (coefs_matrix[column[0]] + (self.mean_m - 1) / self.mean_m * coefs_matrix[column[1]] + (
                (self.mean_m - 1) * (self.mean_m - 2)) / self.mean_m ** 2 * coefs_matrix[column[2]]).to_numpy()

    # метод расчёта интегралов возмущения - протестировано
    def calc_integral(self, coefs: np.array):
        array_I = np.array([])
        for i in range(len(coefs)):
            array_I = np.append(array_I, coefs[i] * self.eta ** i)
        return sum(array_I)

    # метод расчёта температурно-зависимого диаметра сегмента - протестировано
    def calc_d(self):
        self.d = np.array(self.sigma) * (1 - 0.12 * np.exp(-3 * np.array(self.eps) / (1 *self.temperature)))

    # метод расчёта кси(0,1,2,3) - протестировано
    def ksi(self, n: int) -> float:
        return (pi / 6) * self.rho * np.sum(np.array(self.x) * np.array(self.m) * np.array([x ** n for x in self.d]))

    # метод комбинированияв смесях - протестировано
    def comb_sigma(self, i: int, j: int) -> float:
        return 0.5 * (self.sigma[i] + self.sigma[j])

    # метод комбинирования в смесях - протестировано
    def comb_eps(self, i: int, j: int) -> float:
        return (1 - self.k[j]) * (self.eps[i] * self.eps[j]) ** 0.5

    # метод расчёта сжимаемости - протестировано
    def calc_c(self) -> float:
        return (1 + self.mean_m * ((8 * self.eta - 2 * self.eta ** 2) / (1 - self.eta) ** 4) + (1 - self.mean_m) * (
                20 * self.eta - 27 * self.eta ** 2 + 12 * self.eta ** 3 - 2 * self.eta ** 4) / (
                        (1 - self.eta) * (2 - self.eta)) ** 2) ** (-1)

    # метод расчёта m2_eps2_sigma3 - протестировано
    def calc_m2_eps2_sigma3(self) -> float:
        m2_eps2_sigma3 = 0
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                m2_eps2_sigma3 += self.x[i] * self.x[j] * self.m[i] * self.m[j]* ((self.comb_eps(i, j) / (1 * self.temperature)) ** 2) * self.comb_sigma(i, j) ** 3
        return m2_eps2_sigma3

    # метод расчёта m2_eps_sigma3 - протестировано
    def calc_m2_eps_sigma3(self) -> float:
        m2_eps_sigma3 = 0
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                m2_eps_sigma3 += self.x[i] * self.x[j] * self.m[i] * self.m[j]* (self.comb_eps(i, j) / (1 * self.temperature)) * self.comb_sigma(i,j) ** 3
        return m2_eps_sigma3

    # метод расчёта остаточной энергии Гельмгольца дисперсионных сил - протестировано
    def calc_alpha_disp(self):

        a = self.transfom_coefs(pd.read_excel('a-b.xlsx'))
        b = self.transfom_coefs(pd.read_excel('a-b.xlsx'))
        I_1 = self.calc_integral(a)
        I_2 = self.calc_integral(b)

        m2_eps_sigma3 = self.calc_m2_eps_sigma3()
        m2_eps2_sigma3 = self.calc_m2_eps2_sigma3()

        return -2 * pi * self.rho * I_1 * m2_eps_sigma3 - pi * self.rho * self.calc_c() * I_2 * m2_eps2_sigma3

    # метод расчёта ост энергии Гельмгольца твёрдых сфер - протестировано
    def calc_alpha_hs(self):
        ksi0, ksi1 = self.ksi(0), self.ksi(1)
        ksi2, ksi3 = self.ksi(2), self.ksi(3)
        return (1 / self.ksi(0)) * (
                (3 * ksi1 * ksi2) / (1 - ksi3) + ksi2 ** 3 / (ksi3 * (1 - ksi3) ** 2) + np.log(1 - ksi3) * (
                    (ksi2 ** 3 / ksi3 ** 3) - ksi0))

    # метод радиальная функция распределения в системе твёрдых сфер - протестировано
    def radial_func_distr(self):
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
        alpha_hard_sphere = self.calc_alpha_hs()
        g = self.radial_func_distr()
        return self.mean_m * alpha_hard_sphere - np.sum(self.x * (np.array(self.m) - 1) * np.log(g.diagonal()))

    # метод расчёта остаточной энергии Гельмгольца - протестировано
    def calc_energy_helmholtz(self):
        self.calc_m()
        self.calc_d()
        self.eta = self.ksi(3)
        return self.calc_alpha_chain() + self.calc_alpha_disp()

    def calc_z(self):
        eta_array = []
        for i in [-2,-1,1,2]:
            eta_array.append(self.eta + i * self.h)
