import torch
from torch import nn, autograd

from model.utils import sequence_mask


class WGANGPLoss(nn.Module):
    def __init__(self, discriminator, lambda_=10):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_ = lambda_

    def forward(self, real_data, real_hiddens, fake_data, fake_hiddens, lens):
        d_real = self.discriminator(real_data, real_hiddens, lens)
        d_fake = self.discriminator(fake_data, fake_hiddens, lens)
        gradient_penalty = self.get_gradient_penalty(real_data, real_hiddens, fake_data, fake_hiddens, lens)
        wasserstein_distance = d_real.mean() - d_fake.mean()
        d_loss = -wasserstein_distance + gradient_penalty
        return d_loss, wasserstein_distance

    def get_gradient_penalty(self, real_data, real_hiddens, fake_data, fake_hiddens, lens):
        batch_size = len(real_data)
        with torch.no_grad():
            alpha = torch.rand((batch_size, 1, 1)).to(real_data.device)
            interpolates_data = alpha * real_data + (1 - alpha) * fake_data
            interpolates_hiddens = alpha * real_hiddens + (1 - alpha) * fake_hiddens
        interpolates_data = autograd.Variable(interpolates_data, requires_grad=True)
        interpolates_hiddens = autograd.Variable(interpolates_hiddens, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates_data, interpolates_hiddens, lens)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=[interpolates_data, interpolates_hiddens],
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True)
        gradients = torch.cat(gradients, dim=-1)
        gradients = gradients.view(len(gradients), -1)
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        gradient_penalty = gradient_penalty.mean() * self.lambda_
        return gradient_penalty
