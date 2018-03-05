from network_descriptor.layertypes.Layer import Layer
from network_descriptor.activationshapes.StdActivationShape import StdActivationShape
from network_descriptor.weightshapes.StdWeightShape import StdWeightShape

class FullyConnectedLayer(Layer):

    def __init__(self, input_shape, output_feature_count):
        batch_size = input_shape.get_count_probes();
        output_shape = StdActivationShape(batch_size, 1, 1, output_feature_count)
        weight_shape = StdWeightShape(input_shape.get_count_total(), output_feature_count)
        super().__init__(input_shape, output_shape, weight_shape)
