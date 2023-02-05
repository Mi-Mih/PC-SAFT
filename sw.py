import numpy as np
from chemicals.rachford_rice import flash_inner_loop


class SW:
    def __init__(self, omega: np.array, T: float, T_cr: np.array, P: float, P_cr: np.array, z: np.array,
                 matrix_c: np.array):
        self.omega = np.array(omega)  # ацентрический фактор без размерности
        self.matrix_c = matrix_c  # коэффы попарного взаимодействия
        self.T = T  # температура K
        self.T_cr = T_cr  # критическая температура K
        self.P = P  # давление Па
        self.P_cr = P_cr  # критическое давление Па
        self.z = z  # компонентный мольный состав смеси
        self.R = 8.31  # универсальная газовая постоянная

        # параметры уравнения состояния
        self.a = np.zeros((len(z)))
        self.b = np.zeros((len(z)))
        self.c = np.zeros((len(z)))
        self.d = np.zeros((len(z)))

        # параметры модели
        self.K = np.zeros((len(z)))  # коэффициент распределения компонентов смеси
        self.x = np.zeros((len(z)))  # жидкая фаза
        self.y = np.zeros((len(z)))  # газовая фаза
        self.T_r = T / T_cr
        # коэффициенты летучести
        self.fugacity_l = np.zeros((len(z)))
        self.fugacity_v = np.zeros((len(z)))

    # формула Вильсона
    def calc_K(self):
        self.K = (np.exp(5.373 * (1 + self.omega) * (1 - self.T_cr / self.T)) * self.P_cr) / self.P

    # вспомогательные коэффициенты
    def solve_betta_eq(self):
        all_roots = np.zeros((len(self.omega)))
        for k in range(len(self.omega)):
            solution = np.roots([(6 * self.omega[k]), 3, 3, -1])
            solution = solution[solution.imag == 0]
            all_roots[k] = sorted(solution[solution > 0])[0]
        return all_roots

    def calc_omega_b(self):
        betta_c = self.solve_betta_eq()
        return betta_c / (3 * (1 + betta_c * self.omega))

    def calc_omega_a(self):
        betta_c = self.solve_betta_eq()
        return (1 - (1 - betta_c) / (3 * (1 + betta_c * self.omega))) ** 3

    # параметры уравнения состояния
    def func_m(self):
        m = np.zeros((len(self.T_r)))
        k_0 = 0.465 + 1.347 * self.omega - 0.528 * self.omega ** 2
        for k in range(len(self.T_r)):
            if self.T_r[k] <= 1:
                m[k] = k_0[k] + ((5 * self.T_r[k] - 3 * k_0[k] - 1) ** 2) / 70
            else:
                m[k] = k_0[k] + (4 - 3 * k_0[k]) / 70
        return m

    def calc_a(self):
        a_c = self.calc_omega_a() * ((self.R * self.T_cr) ** 2) / self.P_cr
        self.a = a_c * (1 + self.func_m() * (1 - np.sqrt(self.T_cr)) ** 2)

    def calc_b(self):
        self.b = self.calc_omega_b() * self.R * self.T_cr / self.P_cr

    def calc_c(self):
        u = 1 + 3 * self.omega
        w = -3 * self.omega
        self.c = (-u * self.b - np.sqrt((u * self.b) ** 2 - 4 * w)) / 2

    def calc_d(self):
        u = 1 + 3 * self.omega
        w = -3 * self.omega
        self.d = (-u * self.b + np.sqrt((u * self.b) ** 2 - 4 * w)) / 2

    # решение Рашфорда-Райса
    def solve_RR(self):
        _, self.x, self.y = flash_inner_loop(list(self.z), list(self.K))

    # расчёты для смесей
    def calc_am(self, flag):
        if flag == 'l':
            array = self.x
        else:
            array = self.y
        glob_sum = 0
        for k in range(len(array)):
            loc_sum = 0
            for j in range(len(array)):
                loc_sum += array[k] * array[j] * (1 - self.matrix_c[k][j]) * (self.a[k] * self.a[j]) ** 0.5
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

    def calc_dm(self, flag: str):
        if flag == 'l':
            array = self.x
        else:
            array = self.y
        return np.sum(array * self.d)

    def calc_Am(self, flag: str):
        return self.calc_am(flag) * self.P / (self.R ** 2 * self.T ** 2)

    def calc_Bm(self, flag: str):
        return self.calc_bm(flag) * self.P / (self.R * self.T)

    def calc_Cm(self, flag: str):
        return self.calc_cm(flag) * self.P / (self.R * self.T)

    def calc_Dm(self, flag: str):
        return self.calc_dm(flag) * self.P / (self.R * self.T)

    # расчёты для компонентов
    def calc_Ai(self):
        return self.a * self.P / (self.R ** 2 * self.T ** 2)

    def calc_Bi(self):
        return self.b * self.P / (self.R * self.T)

    def calc_Ci(self):
        return self.c * self.P / (self.R * self.T)

    def calc_Di(self):
        return self.d * self.P / (self.R * self.T)

    def calc_fugacity_coeffs(self, flag):
        real_roots = []

        finding_z_factor = [1,
                            self.calc_Cm(flag) + self.calc_Dm(flag) - self.calc_Bm(flag) - 1,
                            self.calc_Am(flag) - self.calc_Bm(flag) * self.calc_Cm(flag) + self.calc_Cm(
                                flag) * self.calc_Dm(flag) - self.calc_Bm(flag) * self.calc_Dm(flag) - self.calc_Dm(
                                flag) - self.calc_Cm(flag),
                            -self.calc_Bm(flag) * self.calc_Cm(flag) * self.calc_Dm(flag) - self.calc_Cm(
                                flag) * self.calc_Dm(flag) - self.calc_Am(flag) * self.calc_Bm(flag)]

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
        for k in range(len(array)):
            for j in range(len(array)):
                matrix_a[k][j] = (1 - self.matrix_c[k][j]) * (self.a[k] * self.a[j]) ** 0.5

        sum = np.zeros((len(self.z)))
        for i in range(len(self.z)):
            sum[i] = np.sum(array * matrix_a[i, :])

        return np.exp(np.log(np.array(array) * self.P) - np.log(Z - self.calc_Bm(flag))
                      - (2 * sum / self.calc_am(flag) - (self.c - self.d) / (
                self.calc_cm(flag) - self.calc_dm(flag))) * np.log(
            (Z + self.calc_Cm(flag)) / (Z + self.calc_Dm(flag))) * self.calc_Am(flag) / (
                              self.calc_Cm(flag) - self.calc_Dm(flag)) \
               + self.calc_Bi() / (Z - self.calc_Bm(flag)) - self.calc_Am(flag) / (
                       self.calc_Cm(flag) - self.calc_Dm(flag)) * (
                       self.calc_Ci() / (Z + self.calc_Cm(flag)) - self.calc_Di() / (Z + self.calc_Dm(flag))))

    def cycle_i(self):
        self.calc_a()
        self.calc_b()
        self.calc_c()
        self.solve_RR()

        self.fugacity_l = self.calc_fugacity_coeffs('l')
        self.fugacity_v = self.calc_fugacity_coeffs('v')
        if np.sum(np.abs(self.fugacity_l / self.fugacity_v - 1) > 10e-5)==len(self.z):
            condition=1
        else:
            condition=0
        return condition

    def launch_SWE(self):
        self.calc_K()
        for s in range(100):
            cond = self.cycle_i()
            if cond:
                break
            else:
                self.K = self.K * self.fugacity_l / self.fugacity_v


if __name__ == '__main__':
    pass
