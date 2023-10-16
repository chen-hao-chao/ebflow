import sys
import argparse

from ebflow.experiments import (twodim, mm)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default='twodim', help='experiment setup')
parser.add_argument('--dataset', type=str, default='sine', help='specification of the dataset')
parser.add_argument('--loss', type=str, default='ssm', help='specification of the loss function')
parser.add_argument('--Mtype', type=str, default='full', help='specification of the matrix type')
parser.add_argument('--restore_path', type=str, default='', help='path of a pretrained models')
parser.add_argument('--eval', action='store_true', help='set True to evaluate the performance')

def main():
    args = parser.parse_args()
    module_name = 'ebflow.experiments.{}'.format(args.config)
    experiment = sys.modules[module_name]
    experiment.main(args)
