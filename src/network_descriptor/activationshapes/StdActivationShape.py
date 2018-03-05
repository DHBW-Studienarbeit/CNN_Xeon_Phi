from network_descriptor.activationshapes.ActivationShape import ActivationShape

class StdActivationShape(ActivationShape):

    def __init__(self, count_probes, count_y, count_x, count_features):
        count_total = count_probes * count_y * count_x * count_features
        super().__init__(count_total, count_probes, count_y, count_x, count_features)

    def get_position(self, p, y, x, feature):
        return feature + x * self._count_features + y * self._count_features * self._count_x + p * self._count_y * self._count_x * self._count_features
