from network_descriptor.layertypes.Layer import Layer
from network_descriptor.activationshapes.ConvActivationShape import ConvActivationShape
from network_descriptor.weightshapes.StdWeightShape import StdWeightShape

class ConvolutionalLayer(Layer):

    def __init__(self, input_shape, filter_size_x, filter_size_y, count_output_features):
        output_shape = ConvActivationShape(input_shape, filter_size_x, filter_size_y, count_output_features)
        input_count = filter_size_x * filter_size_y * input_shape.get_count_features()
        weight_shape = StdWeightShape(input_count, count_output_features)
        super().__init__(input_shape, output_shape, weight_shape)

    def apply_consts(self, act_in_off, act_out_off, weight_off):
        self._act_in_off = act_in_off
        self._act_out_off = act_out_off
        self._weight_off = weight_off
