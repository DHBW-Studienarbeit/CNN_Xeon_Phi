from network_descriptor.layertypes.Layer import Layer
from network_descriptor.activationshapes.StdActivationShape import StdActivationShape
from network_descriptor.weightshapes.StdWeightShape import StdWeightShape

class FullyConnectedLayer(Layer):

    def __init__(self, input_shape, output_feature_count):
        batch_size = input_shape.get_count_probes()
        output_shape = StdActivationShape(batch_size, 1, 1, output_feature_count)
        weight_shape = StdWeightShape(input_shape.get_count_total()//batch_size, output_feature_count)
        super().__init__(input_shape, output_shape, weight_shape)

    def apply_consts(self, act_in_off, act_out_off, weight_off):
        self._act_in_off = act_in_off
        self._act_out_off = act_out_off
        self._weight_off = weight_off
        self._bias_off = weight_off + self.get_weight_shape().get_count_output() * self.get_weight_shape().get_count_input()

    def get_C_typename(self):
        return "FullyConnectedLayer_"

    def get_C_forwardname(self, first):
        if first == True:
            return "layer_fullcon_first_forward"
        else:
            return "layer_fullcon_forward"

    def get_C_backwardname(self, first):
        if first == True:
            return "layer_fullcon_first_backward"
        else:
            return "layer_fullcon_backward"

    def get_C_outputname(self):
        return "layer_fullcon_get_output"
