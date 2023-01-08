import numpy as np
from chemicals.rachford_rice import flash_inner_loop


class MARTIN:
    def __init__(self, omega: float, T: float, T_cr: np.array, P: float, P_cr: np.array, z: np.array, c: np.array):
        self.omega = omega
        self.c = c
        self.T = T
        self.T_cr = T_cr
        self.P = P
        self.P_cr = P_cr
        self.z = z
        self.R = 1  # исправить

        # параметры уравнения состояния
        self.a = None
        self.b = None
        self.c = None

        # параметры модели
        self.K = None
        self.x = None
        self.y = None
        self.T_r = T / T_cr

    # формула Вильсона
    def calc_K(self):
        self.K = (np.exp(5.373 * (1 + self.omega) * (1 - self.T_cr / self.T)) * self.P_cr) / self.P

    # вспомогательные коэффициенты
    def calc_omega_1(self):
        return 0.00756 + 0.90984 * self.omega + 0.1622 * self.omega ** 2 + 0.14549 * self.omega ** 3

    def calc_gamma_0(self):
        return 4.275051 - 8.878889 / self.T_r + 8.508932 / self.T_r ** 2 - 3.481408 / self.T_r ** 3 + 0.576312 / self.T_r ** 4

    def calc_gamma_1(self):
        return 12.856404 - 34.744125 / self.T_r + 37.433095 / self.T_r ** 2 - 18.059421 / self.T_r ** 3 + 3.51405 / self.T_r ** 4

    def calc_a_0(self):
        return -0.1514 * self.T_r + 0.7895 + 0.3314 / self.T_r + 0.029 / self.T_r ** 2 + 0.0015 / self.T_r ** 3

    def calc_a_1(self):
        return -0.237 * self.T_r - 0.7846 / self.T_r + 1.0026 / self.T_r ** 2 + 0.019 / self.T_r ** 3

    # параметры уравнения состояния
    def calc_a(self):
        self.a = 27 * self.R ** 2 * self.T_cr ** 2 * (64 * self.P_cr) * (
                self.calc_a_0() + self.calc_omega_1() * self.calc_a_1())

    def calc_c(self):
        self.c = (
                         0.043 * self.calc_gamma_0() + 0.0713 * self.calc_omega_1() * self.calc_gamma_1()) * self.R * self.T_cr / self.P_cr

    def calc_b(self):
        self.b = (0.082 - 0.0713 * self.calc_omega_1()) * self.R * self.T_cr / self.P_cr

    # решение Рашфорда-Райса
    def solve_RR(self):
        _, self.x, self.y = flash_inner_loop(self.z, self.K)

    '''
    def calc_liquid_z(self):
        self.x = self.z / (self.V*(self.K-1)+1)
    def calc_vapour_z(self):
        self.y = self.z * self.K/(self.V*(self.K-1)+1)
    '''

    # расчёты для смесей
    def calc_am(self):
        glob_sum = 0
        for i in range(len(self.y)):
            loc_sum = 0
            for j in range(len(self.y)):
                loc_sum += self.y[i] * self.y[j] * (1 - self.c[i][j]) * (self.a[i] * self.a[j]) ** 0.5
            glob_sum += loc_sum
        return glob_sum

    def calc_bm(self):
        return np.sum(self.y * self.b)
    def calc_cm(self):
        return np.sum(self.y * self.c)

    def calc_Am(self):
        return self.calc_am() * self.P/(self.R**2 * self.T**2)
    def calc_Bm(self):
        return self.calc_bm() * self.P/(self.R * self.T)
    def calc_Cm(self):
        return self.calc_cm() * self.P/(self.R * self.T)

    # расчёты для компонентов
    def calc_Ai(self):
        return self.calc_a() * self.P/(self.R**2 * self.T**2)
    def calc_Bi(self):
        return self.calc_b() * self.P/(self.R * self.T)
    def calc_Ci(self):
        return self.calc_c() * self.P/(self.R * self.T)