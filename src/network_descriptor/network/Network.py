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
        act_pos_in = 0
        act_mem_i_pos = 0;
        weight_f_pos = 0
        weight_i_pos = 0
        first_run = True
        for current in self._layers:
            if first_run == True:
                # first input is not stored in activations
                act_pos_out = 0
                first_run = False
            else:
                act_pos_out = act_pos_in + current.get_input_shape().get_count_total()
            if current.__class__.__name__ == "MaxPoolingLayer":
                current.apply_consts(act_pos_in, act_pos_out, weight_i_pos, act_mem_i_pos)
                weight_i_pos = weight_i_pos + current.get_weight_shape().get_count_total()
                act_mem_i_pos = act_mem_i_pos + current.get_weight_shape().get_count_output()
            else:
                current.apply_consts(act_pos_in, act_pos_out, weight_f_pos)
                weight_f_pos = weight_f_pos + current.get_weight_shape().get_count_total()
            # next layers input is current layers output
            act_pos_in = act_pos_out
        self._activation_size = act_pos_in
        self._act_mem_i_size = act_mem_i_pos
        self._weights_f_size = weight_f_pos
        self._weights_i_size = weight_i_pos

    def get_input_shape(self):
        return self._input_shape

    def get_output_shape(self):
        return self._output_shape
