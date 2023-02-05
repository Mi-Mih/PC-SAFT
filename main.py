from pc_saft import PCSAFT, launch_pcsaft
from martin import MARTIN
from sw import SW
import numpy as np


def main():
    martin_eq = MARTIN(omega=[0.3, 0.3,0.3], T=400, T_cr=np.array([190.56, 305.32, 369.89]), P=4,
                       P_cr=np.array([4.5992, 4.8722, 4.2512]), z=np.array([0.7, 0.2, 0.1]),
                       matrix_c=np.array([[0, 0.005, 0.01],
                                          [0.005, 0, 0.005],
                                          [0.01, 0.005, 0]]))
    #martin_eq.launch_MartinE()

    sw_eq = SW(omega=[0.3, 0.3, 0.3], T=200, T_cr=np.array([190.56, 305.32, 369.89]), P=4,
               P_cr=np.array([4.5992, 4.8722, 4.2512]), z=np.array([0.7, 0.2, 0.1]),
               matrix_c=np.array([[0, 0.005, 0.01],
                                  [0.005, 0, 0.005],
                                  [0.01, 0.005, 0]]))
    #sw_eq.launch_SWE()

    liquid_state = {'x': [0.1000, 0.3000, 0.6000],
                    'm': [1, 1.6069, 2.002],
                    'sigma': [3.7039, 3.5206, 3.6184],
                    'temperature': 233.15,
                    'rho': 1.18e-2,
                    'eps': [150.03 * 1.380649e-23, 191.42 * 1.380649e-23, 208.11 * 1.380649e-23],
                    'k': [[0.00e+00, 3.00e-04, 1.15e-02],
                          [3.00e-04, 0.00e+00, 5.10e-03],
                          [1.15e-02, 5.10e-03, 0.00e+00]]
                    }

    vapour_state = {'x': [0.1000, 0.3000, 0.6000],
                    'm': [1, 1.6069, 2.002],
                    'sigma': [3.7039, 3.5206, 3.6184],
                    'temperature': 233.15,
                    'rho': 2.37e-12,
                    'eps': [150.03 * 1.380649e-23, 191.42 * 1.380649e-23, 208.11 * 1.380649e-23],
                    'k': [[0.00e+00, 3.00e-04, 1.15e-02],
                          [3.00e-04, 0.00e+00, 5.10e-03],
                          [1.15e-02, 5.10e-03, 0.00e+00]]}

    state = launch_pcsaft(liquid_state, vapour_state)
    print(state)


if __name__ == '__main__':
    main()
