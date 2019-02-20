#!/usr/bin/env python
import argparse
from sagan import *


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--use_generic_dataset', nargs='?', default=False, type=bool)
parser.add_argument('--dataset', nargs='?', default='geoPose3K_final_publish', type=str)
parser.add_argument('--disc_max_filts', nargs='?', default=512, type=int)
parser.add_argument('--disc_min_filts', nargs='?', default=128, type=int)
parser.add_argument('--gen_stretch_z_filts', nargs='?', default=1024, type=int)
parser.add_argument('--gen_max_filts', nargs='?', default=512, type=int)
parser.add_argument('--gen_min_filts', nargs='?', default=128, type=int)
parser.add_argument('--disc_layers', nargs='?', default=5, type=int)
parser.add_argument('--attention', nargs='?', default=True, type=bool)
parser.add_argument('--lr_disc', nargs='?', default=.0004, type=float)
parser.add_argument('--lr_gen', nargs='?', default=.0001, type=float)
parser.add_argument('--z_size', nargs='?', default=128, type=int)
parser.add_argument('--res', nargs='?', default=128, type=int)
parser.add_argument('--train_epoch', nargs='?', default=600000, type=int)
parser.add_argument('--batch_size', nargs='?', default=8, type=int)
parser.add_argument('--save_every', nargs='?', default=1, type=int)
parser.add_argument('--save_img_every', nargs='?', default=1, type=int)
parser.add_argument('--loader_workers', nargs='?', default=4, type=int)
parser.add_argument('--data_perc', nargs='?', default=1, type=float)
parser.add_argument('--save_root', nargs='?', default='austria', type=str)
parser.add_argument('--workers', nargs='?', default=16, type=int)
parser.add_argument('--load_state', nargs='?', type=str)

params = vars(parser.parse_args())

# if load_state arg is not used, then train model from scratch
if __name__ == '__main__':
    sg = Sagan(params)
    if params['load_state']:
        sg.load_state(params['load_state'])
    else:
        print('Starting From Scratch')
    sg.train()
