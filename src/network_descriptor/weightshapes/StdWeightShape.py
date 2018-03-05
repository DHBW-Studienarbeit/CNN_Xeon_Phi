from network_descriptor.weightshapes.WeightShape import WeightShape

class StdWeightShape(WeightShape):

    def __init__(self, count_input, count_output):
        super().__init__( (count_input+1) * count_output )
        self._count_input = count_input
        self._count_output = count_output

    def get_count_output(self):
        return self._count_output

    def get_count_input(self):
        return self._count_input
