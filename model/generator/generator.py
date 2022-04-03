import torch

from model.base_model import BaseModel
from model.utils import sequence_mask
from .generator_layers import GRU, SmoothCondition


class Generator(BaseModel):
    def __init__(self, code_num, hidden_dim, attention_dim, max_len, device=None):
        super().__init__(param_file_name='generator.pt')
        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.max_len = max_len
        self.device = device

        self.noise_dim = hidden_dim
        self.gru = GRU(code_num, hidden_dim, max_len, device)
        self.smooth_condition = SmoothCondition(code_num, attention_dim)

    def forward(self, target_codes, lens, noise):
        samples, hiddens = self.gru(noise)
        samples = self.smooth_condition(samples, lens, target_codes)
        return samples, hiddens

    def sample(self, target_codes, lens, noise=None, return_hiddens=False):
        if noise is None:
            noise = self.get_noise(len(lens))
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
            prob, hiddens = self.forward(target_codes, lens, noise)
            samples = torch.bernoulli(prob).to(prob.dtype)
            samples *= mask
            if return_hiddens:
                hiddens *= mask
                return samples, hiddens
            else:
                return samples

    def get_noise(self, batch_size):
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        return noise

    def get_target_codes(self, batch_size):
        codes = torch.randint(low=0, high=self.code_num, size=(batch_size, ))
        return codes
