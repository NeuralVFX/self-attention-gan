import time
import math
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import helpers as helper
from util import loaders as load
from models import networks as n
from torch.autograd import Variable


############################################################################
# Train
############################################################################


class Sagan:
    """
    Example usage if not using command line:
    from sagan import *
    params = {'dataset': 'geoPose3K_final_publish',
              'batch_size': 8,
              'workers': 16,
              'res': 128,
              'lr_disc': .0004,
              'lr_gen': .0001,
              'gen_max_filts': 512,
              'gen_min_filts': 128,
              'disc_max_filts': 512,
              'disc_min_filts': 128,
              'gen_stretch_z_filts': 1024,
              'z_size': 128,
              'disc_layers': 5,
              'train_epoch': 200,
              'save_every': 10,
              'save_img_every': 10,
              'attention': True,
              'data_perc': 1,
              'save_root': 'austria'}

    sg = Sagan(params)
    sg.train()


    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0
        self.preview_noise = helper.new_random_z(16, params['z_size'], seed=3)

        self.transform = load.NormDenorm([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.train_data = load.MountainDataset(params['dataset'],
                                               self.transform,
                                               output_res=params["res"],
                                               perc=params['data_perc'])

        self.datalen = self.train_data.__len__()

        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=params["batch_size"],
                                                        num_workers=params["workers"],
                                                        shuffle=True,
                                                        drop_last=True)

        print('Data Loader Initialized: ' + str(self.datalen) + ' Images')

        self.model_dict["G"] = n.Generator(layers=int(math.log(params["res"], 2) - 3),
                                           filts=params["gen_stretch_z_filts"],
                                           max_filts=params["gen_max_filts"],
                                           min_filts=params["gen_min_filts"],
                                           attention=params["attention"])

        self.model_dict["D"] = n.Discriminator(channels=3,
                                               layers=params["disc_layers"],
                                               filts_min=params["disc_min_filts"],
                                               filts=params["disc_max_filts"],
                                               attention=params["attention"])

        for i in self.model_dict.keys():
            self.model_dict[i].apply(helper.weights_init_normal)
            self.model_dict[i].cuda()
            self.model_dict[i].train()

        print('Networks Initialized')

        self.opt_dict["G"] = optim.Adam(self.model_dict["G"].parameters(), lr=params['lr_gen'])
        self.opt_dict["D"] = optim.Adam(self.model_dict["D"].parameters(), lr=params['lr_disc'])

        print('Optimizers Initialized')

        # setup history storage #
        self.losses = ['G_Loss', 'D_Loss']
        self.loss_batch_dict = {}
        self.loss_epoch_dict = {}
        self.train_hist_dict = {}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []

    def load_state(self, filepath):
        # Load previously saved sate from disk, including models, optimizers and history
        state = torch.load(filepath)
        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1
        for i in self.model_dict.keys():
            print(f'loading weights:{i}')
            self.model_dict[i].load_state_dict(state['models'][i])
        for i in self.opt_dict.keys():
            print(f'loading opt:{i}')
            self.opt_dict[i].load_state_dict(state['optimizers'][i])
        self.train_hist_dict = state['train_hist']

    def save_state(self, filepath):
        # Save current state of all models, optimizers and history to disk
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()

        model_state = {'iter': self.current_iter,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict}
        torch.save(model_state, filepath)
        return 'Saving State at Iter:' + str(self.current_iter)

    def display_history(self):
        # Draw history of losses, called at end of training
        fig = plt.figure()
        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/' + str(self.params["save_root"]) + '_loss.jpg')
        plt.show()
        plt.close(fig)

    def set_grad(self, model, grad):
        for param in self.model_dict[model].parameters():
            param.requires_grad = grad

    def train_disc(self, fake, real):
        self.opt_dict["D"].zero_grad()
        # discriminate fake samples
        d_result_fake = self.model_dict["D"](fake)
        # discriminate real samples
        d_result_real = self.model_dict["D"](real)

        # add up disc loss and step
        self.loss_batch_dict['D_Loss'] = nn.ReLU()(1.0 - d_result_real).mean() + nn.ReLU()(1.0 + d_result_fake).mean()
        self.loss_batch_dict['D_Loss'].backward()
        self.opt_dict["D"].step()

    def train_gen(self):
        self.opt_dict["G"].zero_grad()

        # feed random Z to generator
        noise = helper.new_random_z(self.params["batch_size"], self.params['z_size'])
        noise_var = Variable(noise)
        fake = self.model_dict["G"](noise_var)

        # get loss from discriminator
        disc_result_fake = self.model_dict["D"](fake)
        self.loss_batch_dict['G_Loss'] = -disc_result_fake.mean()

        # add up gen loss and step
        self.loss_batch_dict['G_Loss'].backward()
        self.opt_dict["G"].step()

        return fake.detach()

    def train_loop(self):
        self.model_dict["G"].train()
        self.set_grad("G", True)
        self.model_dict["D"].train()
        self.set_grad("D", True)

        for loss in self.losses:
            self.loss_epoch_dict[loss] = []

        # Set learning rate
        self.opt_dict["G"].param_groups[0]['lr'] = self.params['lr_gen']
        self.opt_dict["D"].param_groups[0]['lr'] = self.params['lr_disc']
        # print LR and weight decay
        print(f"Sched Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
        [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}") for opt in self.opt_dict.keys()]

        for (real_data) in tqdm(self.train_loader):
            real = real_data.cuda()

            # TRAIN GEN
            self.set_grad("G", True)
            self.set_grad("D", False)
            fake = self.train_gen()

            # TRAIN DISC
            self.set_grad("G", False)
            self.set_grad("D", True)
            self.train_disc(fake, real)

            # append all losses in loss dict
            [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].item()) for loss in self.losses]
            self.current_iter += 1
        [self.train_hist_dict[loss].append(helper.mft(self.loss_epoch_dict[loss])) for loss in self.losses]

    def train(self):
        # Train
        params = self.params
        for epoch in range(params["train_epoch"]):
            epoch_start_time = time.time()

            # TRAIN LOOP
            self.train_loop()

            # save preview image, or weights and loss graph
            if self.current_epoch % params['save_img_every'] == 0:
                helper.show_test(self.model_dict['G'], Variable(self.preview_noise), self.transform,
                                 save='output/' + str(params["save_root"]) + '_' + str(self.current_epoch) + '.jpg')
            if self.current_epoch % params['save_every'] == 0:
                self.display_history()
                save_str = self.save_state(
                    'output/' + str(params["save_root"]) + '_' + str(self.current_epoch) + '.json')
                print(save_str)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print(f'Epoch Training Training Time: {per_epoch_ptime}')
            [print(f'Train {loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]
            print('\n')
            self.current_epoch += 1

        self.display_history()
        print('Hit End of Learning Schedule!')
