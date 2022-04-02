from torch import nn

from model.utils import sequence_mask


class PredictNextLoss(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, input_, label, lens):
        mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)
        loss = self.loss_fn(input_, label)
        loss = loss * mask
        loss = loss.sum(dim=-1).sum(dim=-1).mean()
        return loss
