from network_descriptor.layertypes.Layer import Layer
from network_descriptor.activationshapes.StdActivationShape import StdActivationShape
from network_descriptor.weightshapes.MaxPoolingWeightShape import MaxPoolingWeightShape

class MaxPoolingLayer(Layer):

    def __init__(self, input_shape, filter_size_x, filter_size_y):
        batch_size = input_shape.get_count_probes()
        num_output_features = input_shape.get_count_features()
        output_size_x = input_shape.get_count_x() // filter_size_x
        output_size_y = input_shape.get_count_x() // filter_size_y
        output_count_total = batch_size * output_size_y * output_size_x * num_output_features
        output_shape = StdActivationShape(batch_size, output_size_x, output_size_y, num_output_features)
        weight_shape = MaxPoolingWeightShape(input_shape.get_count_total(), output_count_total, filter_size_x*filter_size_y)
        super().__init__(input_shape, output_shape, weight_shape)
        self._filter_size_x = filter_size_x
        self._filter_size_y = filter_size_y

    def apply_consts(self, act_in_off, act_out_off, weight_off, act_mem_i_off):
        self._act_in_off = act_in_off
        self._act_out_off = act_out_off
        self._weight_off = weight_off
        self._act_mem_i_off = act_mem_i_off

    def get_C_typename(self):
        return "MaxPoolingLayer_"

    def get_C_forwardname(self, first):
        if first == True:
            return "layer_maxpool_first_forward"
        else:
            return "layer_maxpool_forward"

    def get_C_backwardname(self, first):
        if first == True:
            return "layer_maxpool_first_backward"
        else:
            return "layer_maxpool_backward"
