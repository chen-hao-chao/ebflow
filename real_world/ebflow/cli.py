# Code modified from https://github.com/akandykeller/SelfNormalizingFlows
import sys
import argparse

from ebflow.experiments import (mnist_glow, mnist_glow_inv)
from ebflow.experiments import (mnist_cnn, mnist_fc, cifar_cnn, cifar_fc)
from ebflow.experiments import (stl_fc, stl_fc_mcmc, celeb_fc, celeb_fc_mcmc)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default='mnist_fc', help='experiment setup')
parser.add_argument('--loss', type=str, default='ssm', help='specification of the loss function')
parser.add_argument('--withoutMaP', action='store_true', help='set True to disable MaP')
parser.add_argument('--restore_path', type=str, default='', help='path of a trained models')

def main():
    args = parser.parse_args()
    module_name = 'ebflow.experiments.{}'.format(args.config)
    experiment = sys.modules[module_name]
    experiment.main(args)
