import json


class VALIDATION:
    def __init__(self, file_path='diplom_1.txt'):
        with open(f'gas_parameters/gas_parameters.txt') as file:
            self.gas_parameters = json.load(file)
        with open(f'gas_parameters/coefficients_of_pair_interaction.txt') as file:
            self.all_coefficients = json.load(file)
        with open(f'structures/{file_path}') as file:
            self.structures = json.load(file)


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

    return d_martin, d_sw, d_pcsaft


if __name__ == '__main__':
    pass
