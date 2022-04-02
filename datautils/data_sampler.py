import torch
import numpy as np


class CodeSampleIter:
    def __init__(self, code, samples, shuffle=True):
        self.code = code
        self.samples = samples

        self.current_index = 0
        self.length = len(samples)
        if shuffle:
            np.random.shuffle(self.samples)

    def __next__(self):
        sample = self.samples[self.current_index]
        self.current_index += 1
        if self.current_index == self.length:
            self.current_index = 0
        return sample


class DataSampler:
    def __init__(self, ehr_data, lens, device=None):
        self.ehr_data = ehr_data
        self.lens = lens
        self.device = device

        self.size = len(ehr_data)
        self.code_samples = self._get_code_sample_map()

    def _get_code_sample_map(self):
        print('building ehr data sampler ...')
        code_sample_map = dict()
        for i, (sample, len_i) in enumerate(zip(self.ehr_data, self.lens)):
            for t in range(len_i):
                visit = sample[t]
                codes = np.where(visit > 0)[0]
                for code in codes:
                    if code not in code_sample_map:
                        code_sample_map[code] = {i}
                    else:
                        code_sample_map[code].add(i)
        code_samples = [None] * len(code_sample_map)
        for code, samples in code_sample_map.items():
            code_samples[code] = CodeSampleIter(code, list(samples))
        return code_samples

    def sample(self, target_codes):
        lines = np.array([next(self.code_samples[code]) for code in target_codes])
        data, lens = self.ehr_data[lines], self.lens[lines]
        data = torch.from_numpy(data).to(self.device, dtype=torch.float)
        lens = torch.from_numpy(lens).to(self.device, torch.long)
        return data, lens


def get_train_sampler(train_loader, device):
    data_sampler = DataSampler(*train_loader.dataset.data, device)
    return data_sampler
