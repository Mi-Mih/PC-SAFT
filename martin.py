import numpy as np
#from chemicals.rachford_rice import flash_inner_loop


class MARTIN:
    def __init__(self, omega: np.array, T: float, T_cr: np.array, P: float, P_cr: np.array, z: np.array,
                 matrix_c: np.array):
        self.omega = np.array(omega)  # ацентрический фактор без размерности
        self.matrix_c = matrix_c  # коэффы попарного взаимодействия
        self.T = T  # температура K
        self.T_cr = np.array(T_cr)  # критическая температура K
        self.P = P  # давление Па
        self.P_cr = np.array(P_cr)  # критическое давление Па
        self.z = z  # компонентный мольный состав смеси
        self.R = 0.00831  # универсальная газовая постоянная

        # параметры уравнения состояния
        self.a = np.zeros((len(z)))
        self.b = np.zeros((len(z)))
        self.c = np.zeros((len(z)))
        self.d = np.zeros((len(z)))

        # параметры модели
        self.K = np.zeros((len(z)))  # коэффициент распределения компонентов смеси
        self.x = np.zeros((len(z)))  # жидкая фаза
        self.y = np.zeros((len(z)))  # газовая фаза
        self.T_r = T / np.array(T_cr)
        # коэффициенты летучести
        self.fugacity_l = np.zeros((len(z)))
        self.fugacity_v = np.zeros((len(z)))

    # формула Вильсона
    def calc_K(self):
        self.K = (np.exp(5.373 * (np.ones((len(self.omega))) + self.omega) * (np.ones((len(self.omega))) - self.T_cr / self.T)) * self.P_cr) / self.P

    # вспомогательные коэффициенты
    def calc_omega_1(self):
        return 0.00756 + 0.90984 * self.omega + 0.1622 * np.power(self.omega, 2) + 0.14549 * np.power(self.omega, 3)

    def calc_gamma_0(self):
        return 4.275051 - 8.878889 / self.T_r + 8.508932 / self.T_r ** 2 - 3.481408 / self.T_r** 3 + 0.576312 / self.T_r ** 4

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
        self.c = (0.043 * self.calc_gamma_0() + 0.0713 * self.calc_omega_1() * self.calc_gamma_1()) * self.R * self.T_cr / self.P_cr

    def calc_b(self):
        self.b = (0.082 - 0.0713 * self.calc_omega_1()) * self.R * self.T_cr / self.P_cr

    def F_phi(self, phi):
        """
        Функция для нахождение значений фазовых концентраций
        :param phi: объем
        :return: значение фазовых концентраций
        """
        F = 0
        for i in range(len(self.K)):
            F += (self.z[i] * (self.K[i] - 1)) / (phi * (self.K[i] - 1) + 1)
        return F

    def bisection(self, a, b, x, eps=0.00001):
        """
        Рекурсивная функция для нахождение нуля функции F_V методом бисекции
        :param a: левая граница
        :param b: правая граница
        :param x: середина отрезка
        :param eps: погрешность
        :return: значение x, при котором значение функции F_V равно 0 с погрешностью eps
        """
        F_phi_x = self.F_phi(x)
        F_phi_a = self.F_phi(a)
        F_phi_b = self.F_phi(b)

        if abs(F_phi_x) < eps:
            return x
        if F_phi_a * F_phi_b > 0:
            raise Exception('Bisection error')

        if F_phi_a * F_phi_x > 0:
            return self.bisection(x, b, (x + b) / 2)
        if F_phi_b * F_phi_x > 0:
            return self.bisection(a, x, (a + x) / 2)

    # решение Рашфорда-Райса
    def solve_RR(self):
        a = 1 / (1 - max(self.K)) + 0.01
        b = 1 / (1 - min(self.K)) - 0.01
        return self.bisection(a, b, (a + b) / 2)

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
        return self.c * self.P / (self.R * self.T)

    def calc_fugacity_coeffs(self, flag, phi):
        if flag == 'l':
            self.x = [self.z[i] / (phi * (self.K[i] - 1) + 1) for i in range(len(self.K))]
        else:
            self.y = [(self.z[i] * self.K[i]) / (phi * (self.K[i] - 1) + 1) for i in range(len(self.K))]

        real_roots = []
        A = self.calc_Am(flag)
        B = self.calc_Bm(flag)
        C = self.calc_Cm(flag)
        finding_z_factor = [1,
                            2*C-B-1,
                            C**2 - 2*B*C - 2*C + A,
                            -B*C**2 - C**2 - A*B]
        all_roots = np.roots(finding_z_factor)
        for value in all_roots:
            if abs(value.imag) < 0.00001 and value.real > 0:
                real_roots.append(value.real)
        if flag == 'l':
            Z = min(real_roots)
        else:
            Z = max(real_roots)

        return np.exp(Z-1+np.log(1/(Z-self.calc_Bm(flag)))-self.calc_Am(flag)/(self.calc_Cm(flag)+Z)) * self.P + 0.001

    def cycle_i(self):
        self.calc_a()
        self.calc_b()
        self.calc_c()
        phi = self.solve_RR()
        self.fugacity_l = self.calc_fugacity_coeffs('l', phi)
        self.fugacity_v = self.calc_fugacity_coeffs('v', phi)
        if np.sum(np.abs(self.fugacity_l / self.fugacity_v - 1) > 10e-5)==len(self.z):
            condition=1
        else:
            condition=0
        return condition

    def launch_MartinE(self):
        self.calc_K()
        for s in range(100):
            cond = self.cycle_i()
            if cond:
                break
            else:
                self.K = self.K * self.fugacity_l / self.fugacity_v



if __name__ == '__main__':
    pass
