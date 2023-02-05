import numpy as np
from chemicals.rachford_rice import flash_inner_loop


class MARTIN:
    def __init__(self, omega: float, T: float, T_cr: np.array, P: float, P_cr: np.array, z: np.array,
                 matrix_c: np.array):
        self.omega = omega  # ацентрический фактор без размерности
        self.matrix_c = matrix_c  # коэффы попарного взаимодействия
        self.T = T # температура K
        self.T_cr = T_cr # критическая температура K
        self.P = P # давление Па
        self.P_cr = P_cr # критическое давление Па
        self.z = z # компонентный мольный состав смеси
        self.R = 8.31 # универсальная газовая постоянная

        # параметры уравнения состояния
        self.a = None
        self.b = None
        self.c = None

        # параметры модели
        self.K = None #коэффициент распределения компонентов смеси
        self.x = None # жидкая фаза
        self.y = None # газовая фаза
        self.T_r = T / T_cr
        # коэффициенты летучести
        self.fugacity_l = 0
        self.fugacity_v = 0

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

    # расчёты для смесей
    def calc_am(self, flag):
        if flag == 'l':
            array = self.x
        else:
            array = self.y
        glob_sum = 0
        for i in range(len(array)):
            loc_sum = 0
            for j in range(len(array)):
                loc_sum += array[i] * array[j] * (1 - self.matrix_c[i][j]) * (self.a[i] * self.a[j]) ** 0.5
            glob_sum += loc_sum
        return glob_sum

    def calc_bm(self, flag: str):
        if flag == 'l':
            array = self.x
        else:
            array = self.y
        return np.sum(array * self.b)

    def calc_cm(self, flag: str):
        if flag == 'l':
            array = self.x
        else:
            array = self.y
        return np.sum(array * self.c)

    def calc_Am(self, flag: str):
        return self.calc_am(flag) * self.P / (self.R ** 2 * self.T ** 2)

    def calc_Bm(self, flag: str):
        return self.calc_bm(flag) * self.P / (self.R * self.T)

    def calc_Cm(self, flag: str):
        return self.calc_cm(flag) * self.P / (self.R * self.T)

    # расчёты для компонентов
    def calc_Ai(self):
        return self.a * self.P / (self.R ** 2 * self.T ** 2)

    def calc_Bi(self):
        return self.b * self.P / (self.R * self.T)

    def calc_Ci(self):
        return self.calc_c() * self.P / (self.R * self.T)

    def calc_fugacity_coeffs(self, flag):
        real_roots = []
        finding_z_factor = [1,
                            self.calc_Cm() - self.calc_Bm - 1, self.calc_Am() - self.calc_Bm() * self.calc_Cm() - self.calc_Cm(),
                            self.calc_Am() * self.calc_Bm()]
        all_roots = np.roots(finding_z_factor)

        for value in all_roots:
            if abs(value.imag) < 0.00001 and value.real > 0:
                real_roots.append(value.real)

        if flag == 'l':
            array = self.x
            Z = min(real_roots)
        else:
            Z = max(real_roots)
            array = self.y
        matrix_a = np.zeros((len(array), len(array)))
        for i in range(len(array)):
            for j in range(len(array)):
                matrix_a[i][j] = (1 - self.matrix_c[i][j]) * (self.a[i] * self.a[j]) ** 0.5

        sum = 0
        for j in range(len(array)):
            sum += array * matrix_a[:, j]

        return np.exp(np.log(array * self.P) - np.log(Z - self.calc_Bm(flag)) \
               - (2 * sum / self.calc_am(flag) - self.c / self.calc_cm(flag)) * np.log(
            (Z + self.calc_Cm(flag)) / Z) * self.calc_am(flag) / (self.calc_Cm(flag))) + self.calc_Bi()/(Z-self.calc_Bm(flag)) \
               - self.calc_Am(flag)/self.calc_Cm(flag) * (self.calc_Ci()/(Z+self.calc_Cm(flag)))

    def launch_MartinE(self):
        iter = 0
        while sum(np.abs(self.fugacity_l/self.fugacity_v - 1) > 10e-5)!=len(self.fugacity_l):
                iter+=1
                if iter>100:
                    break
                self.calc_a()
                self.calc_b()
                self.calc_c()
                self.calc_K()
                self.solve_RR()
                self.fugacity_l = self.calc_fugacity_coeffs('l')
                self.fugacity_v = self.calc_fugacity_coeffs('v')
                self.K = self.K * self.fugacity_l / self.fugacity_v

if __name__ == '__main__':
    pass
