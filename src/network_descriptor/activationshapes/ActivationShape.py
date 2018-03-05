
class ActivationShape:

    def __init__(self, count_total, count_probes, count_y, count_x, count_features):
        self._count_total = count_total
        self._count_x = count_x
        self._count_y = count_y
        self._count_p = count_probes
        self._count_features = count_features

    def get_count_total(self):
        return self._count_total

    def get_count_probes(self):
        return self._count_p

    def get_count_x(self):
        return self._count_x

    def get_count_y(self):
        return self._count_y

    def get_count_features(self):
        return self._count_features

    def get_position(self, p, y, x, feature):
        raise Error("called get_position() on ActivationShape() which does not implement this method")
