import torch
from torch import nn

from model.base_model import BaseModel
from model.utils import sequence_mask


class Critic(BaseModel):
    def __init__(self, code_num, hidden_dim, generator_hidden_dim, max_len):
        super().__init__(param_file_name='critic.pt')

        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.max_len = max_len

        self.linear = nn.Sequential(
            nn.Linear(code_num + generator_hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, hiddens, lens):
        output = torch.cat([x, hiddens], dim=-1)
        output = self.linear(output).squeeze(dim=-1)

        mask = sequence_mask(lens, self.max_len)
        output = output * mask
        output = output.sum(dim=-1)
        output = output / lens
        return output
