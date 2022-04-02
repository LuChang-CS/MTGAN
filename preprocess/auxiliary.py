import numpy as np

from preprocess.parse_csv import EHRParser


def generate_code_code_adjacent(pids, patient_admission, admission_codes_encoded, code_num):
    print('generating code code adjacent matrix ...')
    n = code_num
    adj = np.zeros((n, n), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        for admission in patient_admission[pid]:
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    adj[c_i, c_j] = 1
                    adj[c_j, c_i] = 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return adj


def real_data_stat(real_data_x, lens):
    admission_num_count = {}
    max_admission_num = 0
    code_visit_count = {}
    code_patient_count = {}
    for patient, len_i in zip(real_data_x, lens):
        if max_admission_num < len_i:
            max_admission_num = len_i
        admission_num_count[len_i] = admission_num_count.get(len_i, 0) + 1

        codes_set = set()
        for i in range(len_i):
            admission = patient[i]
            codes = np.where(admission > 0)[0]
            codes_set.update(codes.tolist())
            for code in codes:
                code_visit_count[code] = code_visit_count.get(code, 0) + 1
        for code in codes_set:
            code_patient_count[code] = code_patient_count.get(code, 0) + 1

    admission_dist = np.zeros((max_admission_num, ))
    for num, count in admission_num_count.items():
        admission_dist[num - 1] = count
    admission_dist /= admission_dist.sum()

    code_visit_dist = np.zeros(len(code_visit_count))
    for code, count in code_visit_count.items():
        code_visit_dist[code] = count
    code_visit_dist /= code_visit_dist.sum()

    code_patient_dist = np.zeros(len(code_patient_count))
    for code, count in code_patient_count.items():
        code_patient_dist[code] = count
    code_patient_dist /= code_patient_dist.sum()
    return admission_dist, code_visit_dist, code_patient_dist


def parse_icd9_range(range_: str):
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    import os
    code_map = {to_standard_icd9(code): cid for code, cid in code_map.items()}
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (0, 0, 0)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    code_level = dict()
    for code, cid in code_map.items():
        three_level_code = code.split('.')[0]
        three_level = three_level_dict[three_level_code]
        code_level[code] = three_level + [cid]

    code_level_matrix = np.zeros((len(code_map), 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]

    return code_level_matrix
