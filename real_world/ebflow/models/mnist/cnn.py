from ebflow.layers import Dequantization, Normalization
from ebflow.layers.transforms import LogitTransform
from ebflow.layers.distributions.uniform import UniformDistribution

from ebflow.layers.flowsequential import FlowSequential, TransSequential
from ebflow.layers.base_distribution import NegativeLogGaussian
from ebflow.layers.linear import Convolution
from ebflow.layers.activations import SmoothLeakyRelu
from ebflow.layers.squeeze import Squeeze

def create_model(num_layers=6, num_blocks=3, kernel_size=7, 
                 data_size=(1, 28, 28),
                 logit_smoothness=1, shift=1e-6, alpha=0.3,
                 bias=True, spec_norm=False, neg_init=True, MaP=True):

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

    block_size = int(num_layers / num_blocks)
    for b in range(num_blocks):
        for l in range(block_size):
            layers.append(Convolution(data_size[0], data_size[0], 
                                      kernel_size=(kernel_size, kernel_size),
                                      bias=bias, stride=1, padding=int((kernel_size-1)/2),
                                      spec_norm=spec_norm, neg_init=neg_init))

            if not (b == num_blocks - 1 and l == block_size - 1):
                layers.append(SmoothLeakyRelu(alpha=alpha, residual=True))

        if not (b == num_blocks - 1):
            layers.append(Squeeze())
            data_size = (data_size[0]*4, data_size[1]//2, data_size[2]//2)

    return FlowSequential(NegativeLogGaussian(size=data_size), *layers), TransSequential(*trans)

