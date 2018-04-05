#include "testsession.h"
#include "network.h"

// initialize network structure based on net mode
/* [[[cog
import cog
from network_descriptor.NetInstance import net

cog.outl("const NeuronalNetwork_t network =")
cog.outl("{")
i=0
for current in net._layers:
    cog.outl(".layer_" + str(i) + " = ")
    cog.outl("{")
    if current.__class__.__name__ == "FullyConnectedLayer":
        cog.outl("// fully connected")
        cog.outl(".input_activation_offset = " + str(current._act_in_off) + ",")
        cog.outl(".input_activation_count = " + str(current.get_input_shape().get_count_total()) + ",")
        cog.outl(".output_activation_offset = " + str(current._act_out_off) + ",")
        cog.outl(".output_activation_count = " + str(current.get_output_shape().get_count_total()) + ",")
        cog.outl(".single_input_count = " + str(current.get_weight_shape().get_count_input()) + ",")
        cog.outl(".single_output_count = " + str(current.get_weight_shape().get_count_output()) + ",")
        cog.outl(".batch_count = " + str(current.get_input_shape().get_count_probes()) + ",")
        cog.outl(".weights_count_total = " + str(current.get_weight_shape().get_count_total()) + ",")
        cog.outl(".weights_offset = " + str(current._weight_off) + ",")
        cog.outl(".biases_offset = " + str(current._bias_off) + ",")
        cog.outl(".weights_start = " + "net_weights_f + " + str(current._weight_off) + ",")
        cog.outl(".biases_start = " + "net_weights_f + " + str(current._bias_off))
    if current.__class__.__name__ == "ConvolutionalLayer":
        cog.outl("// convolutional")
        cog.outl(".input_activation_offset = " + str(current._act_in_off) + ",")
        cog.outl(".input_activation_count = " + str(current.get_input_shape().get_count_total()) + ",")
        cog.outl(".output_activation_offset = " + str(current._act_out_off) + ",")
        cog.outl(".output_activation_count = " + str(current.get_output_shape().get_count_total()) + ",")
        cog.outl(".filter_feature_input_count = " + str(current.get_input_shape().get_count_features()) + ",")
        cog.outl(".filter_feature_input_count = " + str(current.get_input_shape().get_count_features()) + ",")
        cog.outl(".filter_x_count = " + str(current.get_output_shape()._filter_size_x) + ",")
        cog.outl(".filter_y_count = " + str(current.get_output_shape()._filter_size_y) + ",")
        cog.outl(".filter_feature_output_count = " + str(current.get_output_shape().get_count_features()) + ",")
        cog.outl(".batch_count = " + str(current.get_input_shape().get_count_probes()) + ",")
        cog.outl(".filter_matrix_width = " + str(current.get_weight_shape().get_count_input()) + ",")
        cog.outl(".input_matrix_height = " + str(current.get_input_shape().get_count_features() * current.get_output_shape()._filter_size_x) + ",")
        cog.outl(".input_matrix_width = " + str(current.get_output_shape()._num_output_section_columns) + ",")
        cog.outl(".input_matrix_toplayer_elements_count = " + str(current.get_input_shape().get_count_x()*current.get_input_shape().get_count_features()) + ",")
        cog.outl(".partial_output_matrix_count = " + str(current.get_output_shape()._output_section_size) + ",")
        cog.outl(".full_output_matrix_width = " + str(current.get_output_shape()._num_output_section_columns * current.get_output_shape()._filter_size_y) + ",")
        cog.outl(".weights_total_count = " + str(current.get_weight_shape()._count_total) + ",")
        cog.outl(".weights_offset = " + str(current._weight_off) + ",")
        cog.outl(".biases_offset = " + str(current._bias_off) + ",")
        cog.outl(".weights_start = " + "net_weights_f + " + str(current._weight_off) + ",")
        cog.outl(".biases_start = " + "net_weights_f + " + str(current._bias_off))
    if current.__class__.__name__ == "MaxPoolingLayer":
        cog.outl("// maxpooling")
        cog.outl(".input_activation_offset = " + str(current._act_in_off) + ",")
        cog.outl(".input_activation_count = " + str(current.get_input_shape().get_count_total()) + ",")
        cog.outl(".output_activation_offset = " + str(current._act_out_off) + ",")
        cog.outl(".output_activation_count = " + str(current.get_output_shape().get_count_total()) + ",")
        cog.outl(".pooling_layout = ")
        cog.outl("{")
        cog.outl(".relevant_entries_count = " + str(current.get_weight_shape().get_count_total()) + ",")
        cog.outl(".num_of_lines = " + str(current.get_weight_shape().get_count_output()) + ",")
        cog.outl(".relevant_columns_per_line = " + str(current._filter_size_x * current._filter_size_y) + ",")
        cog.outl(".relevant_columns = " + "net_weights_i + " + str(current._weight_off))
        cog.outl("},")
        cog.outl(".weight_shape = ")
        cog.outl("{")
        cog.outl(".relevant_entries_count = " + str(current.get_weight_shape().get_count_output()) + ",")
        cog.outl(".relevant_columns_offset = " + str(current._act_mem_i_off))
        cog.outl("}")
    cog.out("}")
    if i < len(net._layers)-1:
        cog.outl(",")
        i=i+1
    else:
        cog.outl("")
cog.outl("};")
]]] */
const NeuronalNetwork_t network =
{
.layer_0 = 
{
// convolutional
.input_activation_offset = 0,
.input_activation_count = 15680,
.output_activation_offset = 0,
.output_activation_count = 933900,
.filter_feature_input_count = 1,
.filter_feature_input_count = 1,
.filter_x_count = 5,
.filter_y_count = 5,
.filter_feature_output_count = 60,
.batch_count = 20,
.filter_matrix_width = 25,
.input_matrix_height = 5,
.input_matrix_width = 3113,
.input_matrix_toplayer_elements_count = 28,
.partial_output_matrix_count = 186780,
.full_output_matrix_width = 15565,
.weights_total_count = 1560,
.weights_offset = 0,
.biases_offset = 1500,
.weights_start = net_weights_f + 0,
.biases_start = net_weights_f + 1500
},
.layer_1 = 
{
// maxpooling
.input_activation_offset = 0,
.input_activation_count = 933900,
.output_activation_offset = 933900,
.output_activation_count = 172800,
.pooling_layout = 
{
.relevant_entries_count = 691200,
.num_of_lines = 172800,
.relevant_columns_per_line = 4,
.relevant_columns = net_weights_i + 0
},
.weight_shape = 
{
.relevant_entries_count = 172800,
.relevant_columns_offset = 0
}
},
.layer_2 = 
{
// convolutional
.input_activation_offset = 933900,
.input_activation_count = 172800,
.output_activation_offset = 1106700,
.output_activation_count = 226400,
.filter_feature_input_count = 60,
.filter_feature_input_count = 60,
.filter_x_count = 5,
.filter_y_count = 5,
.filter_feature_output_count = 80,
.batch_count = 20,
.filter_matrix_width = 1500,
.input_matrix_height = 300,
.input_matrix_width = 566,
.input_matrix_toplayer_elements_count = 720,
.partial_output_matrix_count = 45280,
.full_output_matrix_width = 2830,
.weights_total_count = 120080,
.weights_offset = 1560,
.biases_offset = 121560,
.weights_start = net_weights_f + 1560,
.biases_start = net_weights_f + 121560
},
.layer_3 = 
{
// maxpooling
.input_activation_offset = 1106700,
.input_activation_count = 226400,
.output_activation_offset = 1333100,
.output_activation_count = 25600,
.pooling_layout = 
{
.relevant_entries_count = 102400,
.num_of_lines = 25600,
.relevant_columns_per_line = 4,
.relevant_columns = net_weights_i + 691200
},
.weight_shape = 
{
.relevant_entries_count = 25600,
.relevant_columns_offset = 172800
}
},
.layer_4 = 
{
// fully connected
.input_activation_offset = 1333100,
.input_activation_count = 25600,
.output_activation_offset = 1358700,
.output_activation_count = 200,
.single_input_count = 1280,
.single_output_count = 10,
.batch_count = 20,
.weights_count_total = 12810,
.weights_offset = 121640,
.biases_offset = 134440,
.weights_start = net_weights_f + 121640,
.biases_start = net_weights_f + 134440
}
};
// [[[end]]]


int main(void)
{
    exec_testsession(network, supplier);
}
