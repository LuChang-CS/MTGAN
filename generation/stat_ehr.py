import numpy as np
from scipy.spatial.distance import jensenshannon as jsd


def get_basic_statistics(data, lens):
    result = data.sum(axis=1).sum(axis=0)

    n_types = (result > 0).sum()
    n_codes = result.sum()
    n_visits = lens.sum()

    avg_code_num = n_codes / n_visits if n_visits > 0 else 0
    avg_visit_num = n_visits / len(data)
    return n_types, n_codes, n_visits, avg_code_num, avg_visit_num


def code_count(data, lens, icode_map):
    count = {}
    for patient, len_i in zip(data, lens):
        for i in range(len_i):
            admission = patient[i]
            codes = np.where(admission > 0)[0]
            for code in codes:
                count[icode_map[code]] = count.get(icode_map[code], 0) + 1
    sorted_count = sorted(count.items(), key=lambda item: item[1], reverse=True)
    return sorted_count


def get_top_k_disease(data, lens, icode_map, code_name_map, top_k=10, file=None):
    count = code_count(data, lens, icode_map)
    print('--------------------------------------------------', file=file)
    sufix = ['0', '00', '1', '01', '2']
    for cid, num in count[:top_k]:
        if cid not in code_name_map:
            for x in sufix:
                if cid + x in code_name_map:
                    cid = cid + x
                    break
        print(code_name_map[cid], ';', num, file=file)
    print('--------------------------------------------------', file=file)
    return count


def normalized_distance(dist1, dist2):
    dist = np.abs(dist1 - dist2) / ((dist1 + dist2) / 2)
    dist = dist.mean()
    return dist


def get_distribution(data, lens, code_num):
    p_count = {}
    v_dist = np.zeros((code_num, ))
    p_dist = np.zeros((code_num,))
    for i, (p, lens) in enumerate(zip(data, lens)):
        for len_i in range(lens):
            d = p[len_i]
            codes = np.where(d > 0)[0]
            for c in codes:
                v_dist[c] += 1
                if c in p_count:
                    p_count[c].add(i)
                else:
                    p_count[c] = {i}
    v_dist /= v_dist.sum()

    for c, s in p_count.items():
        p_dist[c] = len(s)
    p_dist /= p_dist.sum()
    return v_dist, p_dist


def calc_distance(real_data, real_lens, fake_data, fake_lens, code_num):
    real_v_dist, real_p_dist = get_distribution(real_data, real_lens, code_num)
    fake_v_dist, fake_p_dist = get_distribution(fake_data, fake_lens, code_num)
    jsd_v = jsd(real_v_dist, fake_v_dist)
    nd_v = normalized_distance(real_v_dist, fake_v_dist)
    jsd_p = jsd(real_p_dist, fake_p_dist)
    nd_p = normalized_distance(real_p_dist, fake_p_dist)
    return jsd_v, jsd_p, nd_v, nd_p
