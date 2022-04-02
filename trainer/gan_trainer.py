from .generator_trainer import GeneratorTrainer
from .critic_trainer import CriticTrainer
from datautils.data_sampler import get_train_sampler
from logger import Logger


class GANTrainer:
    def __init__(self, args,
                 generator, critic, base_gru,
                 train_loader, test_loader,
                 len_dist, code_map, code_name_map,
                 records_path, params_path):

        self.generator = generator
        self.critic = critic
        self.base_gru = base_gru
        self.params_path = params_path
        self.test_loader = test_loader

        self.g_trainer = GeneratorTrainer(generator, critic,
                                          batch_size=args.batch_size, train_num=args.g_iter,
                                          lr=args.g_lr, betas=(args.betas0, args.betas1),
                                          decay_step=args.decay_step, decay_rate=args.decay_rate)
        self.d_trainer = CriticTrainer(critic, generator, base_gru,
                                       batch_size=args.batch_size, train_num=args.c_iter,
                                       lr=args.c_lr, lambda_=args.lambda_, betas=(args.betas0, args.betas1),
                                       decay_step=args.decay_step, decay_rate=args.decay_rate)
        self.logger = Logger(records_path, generator, code_map, code_name_map, len_dist, train_loader.size, args.save_batch_size)

        self.test_freq = args.test_freq
        self.save_freq = args.save_freq
        self.device = generator.device
        self.iters = args.iteration
        self.train_sampler = get_train_sampler(train_loader, self.device)
        self.batch_size = train_loader.batch_size

    def train(self):
        for i in range(1, self.iters + 1):
            target_codes = self.generator.get_target_codes(self.batch_size)
            real_data, real_lens = self.train_sampler.sample(target_codes)

            d_loss, w_distance = self.d_trainer.step(real_data, real_lens, target_codes)
            g_loss = self.g_trainer.step(target_codes, real_lens)

            self.logger.add_train_point(d_loss, g_loss, w_distance)
            if i % self.test_freq == 0:
                test_d_loss = self.d_trainer.evaluate(self.test_loader, self.device)
                self.logger.add_test_point(test_d_loss)
                line = '{} / {} iterations: D_Loss -- {:.6f} -- G_Loss -- {:.6f} -- W_dist -- {:.6f} -- Test_D_Loss -- {:6f}' \
                    .format(i, self.iters, d_loss, g_loss, w_distance, test_d_loss)
                print('\r' + line)

                self.logger.add_log(line)
                self.logger.plot_train()
                self.logger.plot_test()
                self.logger.stat_generation()
            else:
                line = '{} / {} iterations: D_Loss -- {:.6f} -- G_Loss -- {:.6f} -- W_dist -- {:.6f}' \
                    .format(i, self.iters, d_loss, g_loss, w_distance)
                print('\r' + line, end='')
            if i % self.save_freq == 0:
                self.generator.save(self.params_path, 'generator.{}.pt'.format(i))
                self.critic.save(self.params_path, 'critic.{}.pt'.format(i))

        self.generator.save(self.params_path)
        self.critic.save(self.params_path)

        self.logger.save()
