import os
import random

import torch
import numpy as np

from config import get_generate_args, get_paths
from model import Generator
from datautils.dataloader import load_code_name_map, load_meta_data
from datautils.dataset import DatasetReal
from generation.generate import generate_ehr, get_required_number
from generation.stat_ehr import get_basic_statistics, get_top_k_disease, calc_distance


def generate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path, _, params_path = get_paths(args)
    len_dist, _, _, _, code_map = load_meta_data(dataset_path)
    code_name_map = load_code_name_map(args.data_path)
    icode_map = {v: k for k, v in code_map.items()}
    code_num = len(code_map)

    dataset_real = DatasetReal(os.path.join(dataset_path, 'standard', 'real_data'))
    len_dist = torch.from_numpy(len_dist).to(device)
    max_len = dataset_real.train_set.data[0].shape[1]

    if args.use_iteration == -1:
        param_file_name = 'generator.pt'
    else:
        param_file_name = 'generator.{}.pt'.format(args.use_iteration)

    generator = Generator(code_num=code_num,
                          hidden_dim=args.g_hidden_dim,
                          attention_dim=args.g_attention_dim,
                          max_len=max_len,
                          device=device).to(device)
    generator.load(params_path, param_file_name)

    fake_x, fake_lens = generate_ehr(generator, args.number, len_dist, args.batch_size)

    """------------------------get statistics------------------------"""
    real_x, real_lens = dataset_real.train_set.data
    print('real data')
    n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(real_x, real_lens)
    print('{} samples -- code types: {} -- code num: {} -- avg code num: {:.4f}, avg visit len: {:.4f}'
          .format(args.number, n_types, n_codes, avg_code_num, avg_visit_num))
    get_top_k_disease(real_x, real_lens, icode_map, code_name_map, top_k=10)

    print('fake data')
    n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(fake_x, fake_lens)
    print('{} samples -- code types: {} -- code num: {} -- avg code num: {:.4f}, avg visit len: {:.4f}'
          .format(args.number, n_types, n_codes, avg_code_num, avg_visit_num))
    get_top_k_disease(fake_x, fake_lens, icode_map, code_name_map, top_k=10)

    jsd_v, jsd_p, nd_v, nd_p = calc_distance(real_x, real_lens, fake_x, fake_lens, code_num)
    print('JSD_v: {:.4f}, JSD_p: {:.4f}, ND_v: {:.4f}, ND_p: {:.4f}'.format(jsd_v, jsd_p, nd_v, nd_p))
    """------------------------get statistics------------------------"""

    get_required_number(generator, len_dist, args.batch_size, args.upper_bound)

    print('saving {} synthetic data...'.format(args.number))
    synthetic_path = os.path.join(args.result_path, 'synthetic_{}.npz'.format(args.dataset))
    np.savez_compressed(synthetic_path, x=fake_x, lens=fake_lens)


if __name__ == '__main__':
    args = get_generate_args()
    generate(args)
