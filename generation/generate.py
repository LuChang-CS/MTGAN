import torch
import numpy as np


def generate_ehr(generator, number, len_dist, batch_size):
    fake_x, fake_lens = [], []
    for i in range(0, number, batch_size):
        n = number - i if i + batch_size > number else batch_size
        target_codes = generator.get_target_codes(n)
        lens = torch.multinomial(len_dist, num_samples=n, replacement=True) + 1
        x = generator.sample(target_codes, lens)

        fake_x.append(x.cpu().numpy())
        fake_lens.append(lens.cpu().numpy())
    fake_x = np.concatenate(fake_x, axis=0)
    fake_lens = np.concatenate(fake_lens, axis=-1)

    return fake_x, fake_lens


def get_required_number(generator, len_dist, batch_size, upper_bound=1e7):
    code_types = torch.zeros(generator.code_num, dtype=torch.bool, device=generator.device)
    rn = 0
    while True:
        n = np.random.randint(low=np.floor(0.5 * batch_size), high=np.floor(1.5 * batch_size))
        rn += n

        lens = torch.multinomial(len_dist, num_samples=n, replacement=True) + 1
        target_codes = generator.get_target_codes(n)
        x = generator.sample(target_codes, lens)

        code_types = torch.logical_or(code_types, x.sum(dim=1).sum(dim=0) > 0)
        total_code_types = code_types.sum()
        print(total_code_types.item(), rn)
        if total_code_types == generator.code_num:
            print('required number to generate all diseases:', rn)
            return
        if rn >= upper_bound:
            print('unable to generate all disease within {} samples'.format(upper_bound))
            return
