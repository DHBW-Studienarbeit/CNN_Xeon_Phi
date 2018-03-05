from network_descriptor.layertypes.FullyConnectedLayer import *
from network_descriptor.layertypes.ConvolutionalLayer import *
from network_descriptor.layertypes.MaxPoolingLayer import *


class Network:

    def __init__(self, input_shape):
        self._layers = []
        self._input_shape = input_shape
        self._last_shape = input_shape

    def add_fullyconnected_layer(self, output_feature_count):
        next_layer = FullyConnectedLayer(self._last_shape, output_feature_count)
        self._last_shape = next_layer.get_output_shape()
        self._layers.append(next_layer)

    def add_convolutional_layer(self, filter_size_x, filter_size_y, count_output_features):
        next_layer = ConvolutionalLayer(self._last_shape, filter_size_x, filter_size_y, count_output_features)
        self._last_shape = next_layer.get_output_shape()
        self._layers.append(next_layer)

    def add_maxpooling_layer(self, filter_size_x, filter_size_y):
        next_layer = MaxPoolingLayer(self._last_shape, filter_size_x, filter_size_y)
        self._last_shape = next_layer.get_output_shape()
        self._layers.append(next_layer)

    def generate(self):
        self._output_shape = self._last_shape

    def get_input_shape(self):
        return self._input_shape

    def get_output_shape(self):
        return self._output_shape
