from pc_saft import launch_pcsaft
from martin import MARTIN
from sw import SW
from validation import prepare_data

def main(file_path='diplom_1.txt'):
    d_martin, d_sw, d_pcsaft = prepare_data(file_path)
    martin_eq = MARTIN(T=400, P=4, **d_martin)

    #martin_eq.launch_MartinE()

    sw_eq = SW(P=4,T=400,**d_sw)
    sw_eq.launch_SWE()
    '''
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
    '''
    '''
    vapour_state = {'x': [0.1000, 0.3000, 0.6000],
                    'm': [1, 1.6069, 2.002],
                    'sigma': [3.7039, 3.5206, 3.6184],
                    'temperature': 233.15,
                    'rho': 2.37e-12,
                    'eps': [150.03 * 1.380649e-23, 191.42 * 1.380649e-23, 208.11 * 1.380649e-23],
                    'k': [[0.00e+00, 3.00e-04, 1.15e-02],
                          [3.00e-04, 0.00e+00, 5.10e-03],
                          [1.15e-02, 5.10e-03, 0.00e+00]]}
    '''
    state = launch_pcsaft(**d_pcsaft, **d_pcsaft)
    print(state)


if __name__ == '__main__':
    main('diplom_1.txt')
