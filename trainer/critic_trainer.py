import torch

from model import WGANGPLoss


class CriticTrainer:
    def __init__(self, critic, generator, base_gru, batch_size, train_num, lr, lambda_, betas, decay_step, decay_rate):
        self.critic = critic
        self.generator = generator
        self.base_gru = base_gru
        self.batch_size = batch_size
        self.train_num = train_num

        self.optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=decay_rate)
        self.loss_fn = WGANGPLoss(critic, lambda_=lambda_)

    def _step(self, real_data, real_lens, target_codes):
        real_hiddens = self.base_gru.calculate_hidden(real_data, real_lens)
        fake_data, fake_hiddens = self.generator.sample(target_codes, real_lens, return_hiddens=True)
        loss, wasserstein_distance = self.loss_fn(real_data, real_hiddens, fake_data, fake_hiddens, real_lens)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), wasserstein_distance.item()

    def step(self, real_data, real_lens, target_codes):
        self.critic.train()
        self.generator.eval()

        loss, w_distance = 0, 0
        for _ in range(self.train_num):
            loss_i, w_distance_i = self._step(real_data, real_lens, target_codes)
            loss += loss_i
            w_distance += w_distance_i
        loss /= self.train_num
        w_distance /= self.train_num
        self.scheduler.step()

        return loss, w_distance

    def evaluate(self, data_loader, device):
        self.critic.train()
        with torch.no_grad():
            loss = 0
            for data in data_loader:
                data, lens = data
                data, lens = data.to(device), lens.to(device)
                hiddens = self.base_gru.calculate_hidden(data, lens)
                loss += self.critic(data, hiddens, lens).mean().item()
            loss = -loss / len(data_loader)
            return loss
