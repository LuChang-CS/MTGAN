import torch
from torch import nn

from model.utils import MaskedAttention


class GRU(nn.Module):
    def __init__(self, code_num, hidden_dim, max_len, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device

        self.gru_cell = nn.GRUCell(input_size=code_num, hidden_size=hidden_dim)
        self.hidden2codes = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )

    def step(self, x, h=None):
        h_n = self.gru_cell(x, h)
        codes = self.hidden2codes(h_n)
        return codes, h_n

    def forward(self, noise):
        codes = self.hidden2codes(noise)
        h = torch.zeros(len(codes), self.hidden_dim, device=self.device)
        samples, hiddens = [], []
        for _ in range(self.max_len):
            samples.append(codes)
            codes, h = self.step(codes, h)
            hiddens.append(h)
        samples = torch.stack(samples, dim=1)
        hiddens = torch.stack(hiddens, dim=1)

        return samples, hiddens


class SmoothCondition(nn.Module):
    def __init__(self, code_num, attention_dim):
        super().__init__()
        self.attention = MaskedAttention(code_num, attention_dim)

    def forward(self, x, lens, target_codes):
        score = self.attention(x, lens)
        score_tensor = torch.zeros_like(x)
        score_tensor[torch.arange(len(x)), :, target_codes] = score
        x = x + score_tensor
        x = torch.clip(x, max=1)
        return x
