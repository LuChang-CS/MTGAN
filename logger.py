import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from generation.generate import generate_ehr
from generation.stat_ehr import get_basic_statistics, get_top_k_disease


class Logger:
    def __init__(self, plot_path, generator, code_map, code_name_map, len_dist, save_number, save_batch_size):
        self.plot_path = plot_path
        self.generator = generator
        self.save_number = save_number
        self.save_batch_size = save_batch_size

        self.plots = {
            'train': {
                'd_loss': {
                    'data': [],
                    'title': 'Discriminator Loss'
                },
                'g_loss': {
                    'data': [],
                    'title': 'Generator Loss'
                },
                'w_distance': {
                    'data': [],
                    'title': 'Wasserstein Distance'
                }
            },
            'test': {
                'test_d_loss': {
                    'data': [],
                    'title': 'Test Discriminator Loss'
                }
            },
            'generate': {
                'gen_code_type': {
                    'data': [],
                    'title': 'Generated Code Type'
                },
                'gen_code_num': {
                    'data': [],
                    'title': 'Generated Code Number'
                },
                'gen_avg_code_num': {
                    'data': [],
                    'title': 'Generated Average Code Number'
                }
            }
        }

        self.device = generator.device

        self.logfile = open(os.path.join(plot_path, 'output.log'), 'w', encoding='utf-8')

        self.code_name_map = code_name_map
        self.icode_map = {v: k for k, v in code_map.items()}
        self.len_dist = len_dist

    def append_point(self, key, loss_type, loss):
        self.plots[key][loss_type]['data'].append(loss)

    def add_train_point(self, d_loss, g_loss, w_distance):
        self.append_point('train', 'd_loss', d_loss)
        self.append_point('train', 'g_loss', g_loss)
        self.append_point('train', 'w_distance', w_distance)

    def add_test_point(self, test_d_loss):
        self.append_point('test', 'test_d_loss', test_d_loss)

    def add_gen_point(self, gen_code_type, gen_code_num, gen_avg_code_num):
        self.append_point('generate', 'gen_code_type', gen_code_type)
        self.append_point('generate', 'gen_code_num', gen_code_num)
        self.append_point('generate', 'gen_avg_code_num', gen_avg_code_num)

    def plot_dict(self, key, x):
        for item in self.plots[key].values():
            y, title = item['data'], item['title']
            plt.clf()
            plt.plot(x, y)
            plt.xlabel('Iteration')
            plt.ylabel(title)
            plt.savefig(os.path.join(self.plot_path, title.replace(' ', '_') + '.png'))

    def plot_train(self):
        points_num = len(self.plots['train']['d_loss']['data'])
        x = np.arange(1, points_num + 1)
        self.plot_dict('train', x)

    def plot_test(self):
        train_points_num = len(self.plots['train']['d_loss']['data'])
        test_points_num = len(self.plots['test']['test_d_loss']['data'])
        step = train_points_num // test_points_num
        x = np.arange(1, test_points_num + 1) * step
        self.plot_dict('test', x)

    def plot_gen(self):
        train_points_num = len(self.plots['train']['d_loss']['data'])
        gen_points_num = len(self.plots['generate']['gen_code_type']['data'])
        step = train_points_num // gen_points_num
        x = np.arange(1, gen_points_num + 1) * step
        self.plot_dict('generate', x)

    def stat_generation(self):
        fake_x, fake_lens = generate_ehr(self.generator, self.save_number, self.len_dist, self.save_batch_size)
        n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(fake_x, fake_lens)
        log = 'Generating {} samples -- code types: {} -- code num: {} -- avg code num: {:.4f}, avg visit len: {:.4f}' \
            .format(self.save_number, n_types, n_codes, avg_code_num, avg_visit_num)
        self.add_log(log)
        print(log)
        get_top_k_disease(fake_x, fake_lens, self.icode_map, self.code_name_map, top_k=10, file=self.logfile)
        self.add_log('\n')
        self.logfile.flush()

        self.add_gen_point(n_types, n_codes, avg_code_num)
        self.plot_gen()

    def add_log(self, line):
        t = type(line)
        if t is str:
            self.logfile.write(line + '\n')
        elif t is list:
            lines = [line_ + '\n' for line_ in line]
            self.logfile.writelines(lines)
        self.logfile.flush()

    def save(self):
        pickle.dump(self.plots, open(os.path.join(self.plot_path, 'history.log'), 'wb'))
