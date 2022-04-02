import torch

from model import PredictNextLoss


class BaseGRUTrainer:
    def __init__(self, args, base_gru, max_len, train_loader, params_path):
        self.base_gru = base_gru
        self.train_loader = train_loader
        self.params_path = params_path

        self.epochs = args.base_gru_epochs

        self.optimizer = torch.optim.Adam(base_gru.parameters(), lr=args.base_gru_lr)
        self.loss_fn = PredictNextLoss(max_len)

    def train(self):
        print('pre-training base gru...')
        for epoch in range(1, self.epochs + 1):
            print('Epoch %d / %d:' % (epoch, self.epochs))
            total_loss = 0.0
            total_num = 0
            steps = len(self.train_loader)
            for step, data in enumerate(self.train_loader, start=1):
                x, lens, y = data
                output = self.base_gru(x)
                loss = self.loss_fn(output, y, lens)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(x)
                total_num += len(x)

                print('\r    Step %d / %d, loss: %.4f' % (step, steps, total_loss / total_num), end='')

            print('\r    Step %d / %d, loss: %.4f' % (steps, steps, total_loss / total_num))
        self.base_gru.save(self.params_path)
