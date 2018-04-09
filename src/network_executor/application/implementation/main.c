#include <stdio.h>

#include "settings.h"
#include "trainsession.h"
#include "testsession.h"
#include "network.h"
#include "weightgenerator.h"

// initialize network structure based on net mode
/* [[[cog
import cog
from network_descriptor.NetInstance import net

cog.outl("NeuronalNetwork_t network =")
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
NeuronalNetwork_t network =
{
.layer_0 =
{
// convolutional
.input_activation_offset = 0,
.input_activation_count = 7840,
.output_activation_offset = 0,
.output_activation_count = 46350,
.filter_feature_input_count = 1,
.filter_feature_input_count = 1,
.filter_x_count = 5,
.filter_y_count = 5,
.filter_feature_output_count = 6,
.batch_count = 10,
.filter_matrix_width = 25,
.input_matrix_height = 5,
.input_matrix_width = 1545,
.input_matrix_toplayer_elements_count = 28,
.partial_output_matrix_count = 9270,
.full_output_matrix_width = 7725,
.weights_total_count = 156,
.weights_offset = 0,
.biases_offset = 150,
.weights_start = net_weights_f + 0,
.biases_start = net_weights_f + 150
},
.layer_1 =
{
// maxpooling
.input_activation_offset = 0,
.input_activation_count = 46350,
.output_activation_offset = 46350,
.output_activation_count = 8640,
.pooling_layout =
{
.relevant_entries_count = 34560,
.num_of_lines = 8640,
.relevant_columns_per_line = 4,
.relevant_columns = net_weights_i + 0
},
.weight_shape =
{
.relevant_entries_count = 8640,
.relevant_columns_offset = 0
}
},
.layer_2 =
{
// convolutional
.input_activation_offset = 46350,
.input_activation_count = 8640,
.output_activation_offset = 54990,
.output_activation_count = 11120,
.filter_feature_input_count = 6,
.filter_feature_input_count = 6,
.filter_x_count = 5,
.filter_y_count = 5,
.filter_feature_output_count = 8,
.batch_count = 10,
.filter_matrix_width = 150,
.input_matrix_height = 30,
.input_matrix_width = 278,
.input_matrix_toplayer_elements_count = 72,
.partial_output_matrix_count = 2224,
.full_output_matrix_width = 1390,
.weights_total_count = 1208,
.weights_offset = 156,
.biases_offset = 1356,
.weights_start = net_weights_f + 156,
.biases_start = net_weights_f + 1356
},
.layer_3 =
{
// maxpooling
.input_activation_offset = 54990,
.input_activation_count = 11120,
.output_activation_offset = 66110,
.output_activation_count = 1280,
.pooling_layout =
{
.relevant_entries_count = 5120,
.num_of_lines = 1280,
.relevant_columns_per_line = 4,
.relevant_columns = net_weights_i + 34560
},
.weight_shape =
{
.relevant_entries_count = 1280,
.relevant_columns_offset = 8640
}
},
.layer_4 =
{
// fully connected
.input_activation_offset = 66110,
.input_activation_count = 1280,
.output_activation_offset = 67390,
.output_activation_count = 100,
.single_input_count = 128,
.single_output_count = 10,
.batch_count = 10,
.weights_count_total = 1290,
.weights_offset = 1364,
.biases_offset = 2644,
.weights_start = net_weights_f + 1364,
.biases_start = net_weights_f + 2644
}
};
// [[[end]]]


DataSupplier_t trainsupplier, testsupplier;


int main(void)
{
    Int_t iteration;
    Float_t test_cost;
    // generate weights for the network
    weightgen_generate(NETWORK_WEIGHTS_F_SIZE, net_weights_f);
    // initialize suppliers to read input data and labels
    datasupply_init(&trainsupplier, CONFIG_NUM_TRAINFILES, CONFIG_DIR_TRAIN);
    datasupply_init(&testsupplier, CONFIG_NUM_TESTFILES, CONFIG_DIR_TEST);
    // test with random weights first; just for later comparison
    test_cost = exec_testsession(&network, &testsupplier, 1);
    dump_weights();
    printf("Initial cost: ");
    printf(FLOAT_T_ESCAPE, test_cost);
    printf("\n");
    // do iterations consisting of training and testing
    for(iteration=1; iteration<=CONFIG_NUM_OF_ITERATIONS; iteration++)
    {
        exec_trainsession(&network, &trainsupplier, CONFIG_TRAININGS_PER_TEST);
        test_cost = exec_testsession(&network, &testsupplier, 1);
        printf("Iteration %d: ", iteration);
        printf(FLOAT_T_ESCAPE, test_cost);
        printf("\n");
    }
    dump_weights();
    return 0;
}
