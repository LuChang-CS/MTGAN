import torch
from torch import nn


def sequence_mask(lengths, max_length):
    t = lengths.view((-1, 1))
    result = torch.arange(max_length).view((1, -1)).to(lengths.device)
    result = result < t
    return result


def div_no_nan(a, b):
    mask = b == 0
    b_p = b + mask
    result = a / b_p
    result = result * (~mask)
    return result


def masked_softmax(x, mask):
    mask = (~mask).to(x.dtype)
    mask[mask == 1] = float('-inf')
    x = x + mask
    # x = x.masked_fill(mask == 0, float('-inf'))
    result = torch.softmax(x, dim=-1)
    return result


class MaskedAttention(nn.Module):
    def __init__(self, code_num, attention_dim):
        super().__init__()
        self.code_num = code_num
        self.attention_dim = attention_dim

        self.w_omega = nn.Linear(code_num, attention_dim)
        self.u_omega = nn.Linear(attention_dim, 1)

    def forward(self, x, lens):
        t = self.w_omega(x)
        vu = self.u_omega(t).squeeze(dim=-1)

        mask = sequence_mask(lens, vu.shape[-1])
        score = masked_softmax(vu, mask)
        return score
