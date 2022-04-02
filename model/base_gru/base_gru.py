import torch
from torch import nn

from model.base_model import BaseModel
from model.utils import sequence_mask


class BaseGRU(BaseModel):
    def __init__(self, code_num, hidden_dim, max_len):
        super().__init__(param_file_name='base_gru.pt')
        self.gru = nn.GRU(input_size=code_num, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )
        self.max_len = max_len

    def forward(self, x):
        outputs, _ = self.gru(x)
        output = self.linear(outputs)
        return output

    def calculate_hidden(self, x, lens):
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
            outputs, _ = self.gru(x)
            output = outputs * mask
            return output
