from network_descriptor.activationshapes.ActivationShape import ActivationShape
from network_descriptor.weightshapes.WeightShape import WeightShape

class Layer:
    def __init__(self, input_shape, output_shape, weight_shape):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._weight_shape = weight_shape

    def get_input_shape(self):
        return self._input_shape

    def get_output_shape(self):
        return self._output_shape

    def get_weight_shape(self):
        return self._weight_shape
