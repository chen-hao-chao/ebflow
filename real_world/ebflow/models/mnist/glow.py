# Code modified from https://github.com/akandykeller/SelfNormalizingFlows
from ebflow.layers import Dequantization, Normalization
from ebflow.layers.transforms import LogitTransform
from ebflow.layers.distributions.uniform import UniformDistribution

from ebflow.layers.flowsequential import FlowSequential, TransSequential
from ebflow.layers.base_distribution import NegativeLogGaussian
from ebflow.layers.linear import Conv1x1, Convolution
from ebflow.layers.squeeze import Squeeze
from ebflow.layers.coupling import Coupling
from ebflow.layers.splitprior import SplitPrior
from ebflow.layers.activations import ActNorm

def create_model(num_blocks=2, block_size=16,
                 actnorm=True, split_prior=True,
                 activation='leakyrelu',
                 data_size=(1, 28, 28),
                 logit_smoothness=1, shift=1e-6, MaP=True,
                 width=512):
    trans = []
    layers = []
    if MaP:
        trans.append(Dequantization(UniformDistribution(size=data_size)))
        trans.append(Normalization(translation=0, scale=256))
        trans.append(Normalization(translation=-shift, scale=1 / (1 - 2 * shift)))
        trans.append(LogitTransform(logit_smoothness))
    else:
        trans.append(Dequantization(UniformDistribution(size=data_size)))
        layers.append(Normalization(translation=0, scale=256))
        layers.append(Normalization(translation=-shift, scale=1 / (1 - 2 * shift)))
        layers.append(LogitTransform(logit_smoothness))

    for l in range(num_blocks):
        layers.append(Squeeze())
        data_size = (data_size[0]*4, data_size[1]//2, data_size[2]//2)

        for k in range(block_size):
            if actnorm:
                layers.append(ActNorm(data_size[0]))
            layers.append(Conv1x1(data_size[0]))
            layers.append(Coupling(data_size, activation=activation, width=width))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(data_size, NegativeLogGaussian))
            data_size = (data_size[0] // 2, data_size[1], data_size[2])

    return FlowSequential(NegativeLogGaussian(size=data_size), *layers), TransSequential(*trans) # ($)
