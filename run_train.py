import random

import torch
import numpy as np

from config import get_training_args, get_paths
from model import Generator, Critic, BaseGRU
from trainer import GANTrainer, BaseGRUTrainer
from datautils.dataloader import load_code_name_map, load_meta_data, get_train_test_loader, get_base_gru_train_loader


def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path, records_path, params_path = get_paths(args)
    len_dist, code_visit_dist, code_patient_dist, code_adj, code_map = load_meta_data(dataset_path)
    code_name_map = load_code_name_map(args.data_path)
    train_loader, test_loader, max_len = get_train_test_loader(dataset_path, args.batch_size, device)

    code_num = len(code_adj)
    len_dist = torch.from_numpy(len_dist).to(device)

    base_gru = BaseGRU(code_num=code_num, hidden_dim=args.g_hidden_dim, max_len=max_len).to(device)
    try:
        base_gru.load(params_path)
    except FileNotFoundError:
        base_gru_trainloader = get_base_gru_train_loader(dataset_path, args.batch_size, device)
        base_gru_trainer = BaseGRUTrainer(args, base_gru, max_len, base_gru_trainloader, params_path)
        base_gru_trainer.train()
    base_gru.eval()

    generator = Generator(code_num=code_num,
                          hidden_dim=args.g_hidden_dim,
                          attention_dim=args.g_attention_dim,
                          max_len=max_len,
                          device=device).to(device)
    critic = Critic(code_num=code_num,
                    hidden_dim=args.c_hidden_dim,
                    generator_hidden_dim=args.g_hidden_dim,
                    max_len=max_len).to(device)

    print('Param number:', count_model_params(generator) + count_model_params(critic))

    trainer = GANTrainer(args,
                         generator=generator, critic=critic, base_gru=base_gru,
                         train_loader=train_loader, test_loader=test_loader,
                         len_dist=len_dist, code_map=code_map, code_name_map=code_name_map,
                         records_path=records_path, params_path=params_path)
    trainer.train()


if __name__ == '__main__':
    args = get_training_args()
    train(args)
