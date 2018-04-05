from network_descriptor.activationshapes.ActivationShape import ActivationShape

class ConvActivationShape(ActivationShape):

    def __init__(self, input_shape, filter_size_x, filter_size_y, count_output_features):
        batch_size = input_shape.get_count_probes();
        output_size_x = input_shape.get_count_x() - filter_size_x + 1
        output_size_y = input_shape.get_count_x() - filter_size_y + 1
        num_output_section_columns = ((input_shape.get_count_probes() * input_shape.get_count_y() - filter_size_y + 1) * input_shape.get_count_x()) // filter_size_x
        output_count_total = num_output_section_columns * count_output_features * filter_size_x
        super().__init__(output_count_total, batch_size, output_size_y, output_size_x, count_output_features)
        self._input_shape = input_shape
        self._filter_size_x = filter_size_x
        self._filter_size_y = filter_size_y
        self._num_output_section_columns = num_output_section_columns
        self._output_section_size = num_output_section_columns * count_output_features

    def get_position(self, p, y, x, feature):
        input_start_position = self._input_shape.get_position(p, y, x, 0) // input_shape.get_count_features()
        output_section_index = input_start_position % self._filter_size_x
        inner_section_column = input_start_position // self._filter_size_x
        return output_section_index * self._output_section_size + inner_section_column * self.get_count_features() + feature
