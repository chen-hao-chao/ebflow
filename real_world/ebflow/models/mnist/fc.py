# Code modified from https://github.com/akandykeller/SelfNormalizingFlows
from functools import reduce

from ebflow.layers import Dequantization, Normalization
from ebflow.layers.transforms import LogitTransform
from ebflow.layers.distributions.uniform import UniformDistribution

from ebflow.layers.flowsequential import FlowSequential, TransSequential
from ebflow.layers.base_distribution import NegativeLogGaussian
from ebflow.layers.linear import FullyConnected
from ebflow.layers.activations import SmoothLeakyRelu

def create_model(num_layers=2, logit_smoothness=1,
                 data_size=(1, 28, 28), shift=1e-6, alpha=0.3,
                 bias=True, neg_init=False, MaP=True):
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

    # Construct model
    size = reduce(lambda x,y: x*y, data_size)
    for l in range(num_layers):
        layers.append(FullyConnected(size, size, bias=bias, neg_init=neg_init))
        if (l+1) < num_layers:
            layers.append(SmoothLeakyRelu(alpha=alpha))

    return FlowSequential(NegativeLogGaussian(size=(size,)), *layers), TransSequential(*trans)
