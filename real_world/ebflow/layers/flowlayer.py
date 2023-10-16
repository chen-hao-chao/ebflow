# Code from https://github.com/akandykeller/SelfNormalizingFlows
from abc import ABCMeta, abstractmethod
import torch.nn as nn

class FlowLayer(nn.Module, metaclass=ABCMeta):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    @abstractmethod
    def forward(self, input, context=None):
        pass

    @abstractmethod
    def reverse(self, input, context=None):
        pass

    @abstractmethod
    def logdet(self, input, context=None):
        pass


class LinearFlowLayer(FlowLayer):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    @abstractmethod
    def forward(self, input, context=None):
        pass

    @abstractmethod
    def reverse(self, input, context=None):
        pass

    @abstractmethod
    def logdet(self, input, context=None):
        pass