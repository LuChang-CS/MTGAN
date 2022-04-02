import os
import pickle

import numpy as np

from preprocess.parse_csv import Mimic3Parser, Mimic4Parser
from preprocess.encode import encode_concept
from preprocess.build_dataset import split_patients
from preprocess.build_dataset import build_code_xy, build_heart_failure_y, build_real_data, build_real_next_xy, build_visit_x
from preprocess.auxiliary import generate_code_code_adjacent, real_data_stat, generate_code_levels
from config import get_preprocess_args


PARSERS = {
    'mimic3': Mimic3Parser,
    'mimic4': Mimic4Parser
}


if __name__ == '__main__':
    args = get_preprocess_args()

    data_path = args.data_path
    dataset = args.dataset  # mimic3 or mimic4

    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()
    parsed_path = os.path.join(dataset_path, 'parsed')
    if args.from_saved:
        patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
        admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
    else:
        parser = PARSERS[dataset](raw_path)
        sample_num = args.sample_num if dataset == 'mimic4' else None
        patient_admission, admission_codes = parser.parse(sample_num)
        print('saving parsed data ...')
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)
        pickle.dump(patient_admission, open(os.path.join(parsed_path, 'patient_admission.pkl'), 'wb'))
        pickle.dump(admission_codes, open(os.path.join(parsed_path, 'admission_codes.pkl'), 'wb'))

    patient_num = len(patient_admission)

    def stat(data):
        lens = [len(item) for item in data.values()]
        max_, min_, avg = max(lens), min(lens), sum(lens) / len(data)
        return max_, min_, avg

    admission_stats = stat(patient_admission)
    visit_code_stats = stat(admission_codes)
    print('patient num: %d' % patient_num)
    print('visit num: %d' % len(admission_codes))
    print('max, min, mean admission num: %d, %d, %.2f' % admission_stats)
    print('max, min, mean visit code num: %d, %d, %.2f' % visit_code_stats)

    max_admission_num = admission_stats[0]

    print('encoding codes ...')
    admission_codes_encoded, code_map = encode_concept(patient_admission, admission_codes)
    code_num = len(code_map)
    print('There are %d codes' % code_num)
    # hf_list = np.array([cid for code, cid in code_map.items() if code.startswith('428')])
    # pk_list = np.array([cid for code, cid in code_map.items() if code.startswith('332')])
    code_levels = generate_code_levels(data_path, code_map)
    pickle.dump({
        'code_levels': code_levels,
    }, open(os.path.join(parsed_path, 'code_levels.pkl'), 'wb'))

    print('splitting training, and test patients')
    train_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map,
        train_num=args.train_num,
        seed=args.seed
    )
    print('There are %d train, %d test samples' % (len(train_pids), len(test_pids)))
    code_adj = generate_code_code_adjacent(pids=train_pids, patient_admission=patient_admission,
                                           admission_codes_encoded=admission_codes_encoded, code_num=code_num)

    common_args = [patient_admission, admission_codes_encoded, max_admission_num, code_num]

    print('build train real data ...')
    (train_real_data_x, train_real_data_lens) = build_real_data(train_pids, *common_args)
    print('build test real data ...')
    (test_real_data_x, test_real_data_lens) = build_real_data(test_pids, *common_args)

    admission_dist, code_visit_dist, code_patient_dist = real_data_stat(train_real_data_x, train_real_data_lens)

    print('build train visit data ...')
    train_visit_x, train_timestep = build_visit_x(train_real_data_x, train_real_data_lens, code_num)

    print('build train real next ...')
    (train_real_next_x, train_real_next_y, train_real_next_lens) = build_real_next_xy(train_real_data_x, train_real_data_lens)

    print('building train codes features and labels ...')
    (train_code_x, train_codes_y, train_visit_lens) = build_code_xy(train_real_data_x, train_real_data_lens)
    print('building test codes features and labels ...')
    (test_code_x, test_codes_y, test_visit_lens) = build_code_xy(test_real_data_x, test_real_data_lens)

    print('building train heart failure labels ...')
    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    print('building test heart failure labels ...')
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)

    print('building train parkinson labels ...')
    train_parkinson_y = build_heart_failure_y('332', train_codes_y, code_map)
    print('building test parkinson labels ...')
    test_parkinson_y = build_heart_failure_y('332', test_codes_y, code_map)

    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(dataset_path, 'standard')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)

    print('saving real data')
    real_data_path = os.path.join(standard_path, 'real_data')
    if not os.path.exists(real_data_path):
        os.makedirs(real_data_path)
    print('\tsaving train real data ...')
    np.savez_compressed(os.path.join(real_data_path, 'train.npz'),
                        x=train_real_data_x, lens=train_real_data_lens)
    print('\tsaving test real data ...')
    np.savez_compressed(os.path.join(real_data_path, 'test.npz'),
                        x=test_real_data_x, lens=test_real_data_lens)

    print('saving visit data')
    visit_path = os.path.join(standard_path, 'single_visits')
    if not os.path.exists(visit_path):
        os.makedirs(visit_path)
    print('\tsaving train visit data ...')
    np.savez_compressed(os.path.join(visit_path, 'train.npz'), x=train_visit_x)
    np.savez_compressed(os.path.join(visit_path, 'train_timestep.npz'), x=train_timestep)

    print('saving real next data')
    real_next_path = os.path.join(standard_path, 'real_next')
    if not os.path.exists(real_next_path):
        os.makedirs(real_next_path)
    print('\tsaving train real next data ...')
    np.savez_compressed(os.path.join(real_next_path, 'train.npz'),
                        x=train_real_next_x, lens=train_real_next_lens, y=train_real_next_y)

    print('saving task data')
    task_path = os.path.join(standard_path, 'real_task')
    if not os.path.exists(task_path):
        os.makedirs(task_path)
        os.mkdir(os.path.join(task_path, 'train'))
        os.mkdir(os.path.join(task_path, 'test'))
    print('\tsaving task training data')
    np.savez_compressed(os.path.join(task_path, 'train', 'feature.npz'), x=train_code_x, lens=train_visit_lens)
    np.savez_compressed(os.path.join(task_path, 'train', 'codes.npz'), y=train_codes_y)
    np.savez_compressed(os.path.join(task_path, 'train', 'hf.npz'), y=train_hf_y)
    np.savez_compressed(os.path.join(task_path, 'train', 'parkinson.npz'), y=train_parkinson_y)
    print('\tsaving task test data')
    np.savez_compressed(os.path.join(task_path, 'test', 'feature.npz'), x=test_code_x, lens=test_visit_lens)
    np.savez_compressed(os.path.join(task_path, 'test', 'codes.npz'), y=test_codes_y)
    np.savez_compressed(os.path.join(task_path, 'test', 'hf.npz'), y=test_hf_y)
    np.savez_compressed(os.path.join(task_path, 'test', 'parkinson.npz'), y=test_parkinson_y)

    np.savez_compressed(os.path.join(standard_path, 'real_data_stat.npz'),
                        admission_dist=admission_dist,
                        code_visit_dist=code_visit_dist,
                        code_patient_dist=code_patient_dist)
    #
    np.savez_compressed(os.path.join(standard_path, 'code_adj.npz'), code_adj=code_adj)
