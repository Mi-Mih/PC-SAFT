import json


class VALIDATION:
    def __init__(self, file_path='diplom_1.txt'):
        with open(f'gas_parameters/gas_parameters.txt') as file:
            self.gas_parameters = json.load(file)
        with open(f'gas_parameters/coefficients_of_pair_interaction.txt') as file:
            self.all_coefficients = json.load(file)
        with open(f'structures/{file_path}') as file:
            self.structures = json.load(file)


def calc_m_eps_sigma(type_el, molar):
    coeffs = {'parafin': {'m': [0.02569, 0.8709], 'm_sigma': [1.72840, 18.787], 'm_eps_k': [6.8248, 141.1400]},
              'naften': {'m': [0.02254, 0.6827], 'm_sigma': [1.7115, 1.9393], 'm_eps_k': [6.4962, 154, 53]},
              'arafen': {'m': [0.02576, 0.2588], 'm_sigma': [1.7539, -21.324], 'm_eps_k': [6.6756, 172.4]}}
    d = {}
    for k in coeffs[type_el].keys():
        d[k] = coeffs[type_el][k][0] * molar + coeffs[type_el][k][1]
    return d


def prepare_data(file_path='diplom_1.txt'):
    v = VALIDATION(file_path)
    d_martin = {'omega': [], 'T_cr': [], 'P_cr': []}
    d_sw = {'omega': [], 'T_cr': [], 'P_cr': []}
    d_pcsaft = {'m': [], 'sigma': [], 'eps': []}

    for i in v.structures.keys():
        for j in d_martin.keys():
            if j not in v.gas_parameters[i].keys():
                continue
            d_martin[j].append(v.gas_parameters[i][j])
            d_sw[j].append(v.gas_parameters[i][j])
        for j in d_pcsaft.keys():
            if j not in v.gas_parameters[i].keys():
                continue
            d_pcsaft[j].append(v.gas_parameters[i][j])
    d_sw['z'] = list(v.structures.values())
    d_martin['z'] = list(v.structures.values())
    d_pcsaft['x'] = list(v.structures.values())

    matrix_c = []
    for i in v.structures.keys():
        coeffs_list = []
        for j in v.structures.keys():
            coeffs_list.append(v.all_coefficients[i][j])
        matrix_c.append(coeffs_list)

    d_martin['matrix_c'] = matrix_c
    d_sw['matrix_c'] = matrix_c
    d_pcsaft['k'] = matrix_c

    d_pcsaft['m'],d_pcsaft['sigma'],d_pcsaft['eps'] = [],[],[]
    for i in v.structures.keys():
        molar = v.gas_parameters[i]['M']
        type_el = v.gas_parameters[i]['type_el']
        d=calc_m_eps_sigma(type_el, molar)
        d_pcsaft['m'].append(d['m'])
        d_pcsaft['sigma'].append((d['m_sigma']/d['m'])**(1/3))
        d_pcsaft['eps'].append(d['m_eps_k']/d['m'])

    return d_martin, d_sw, d_pcsaft


if __name__ == '__main__':
    pass
