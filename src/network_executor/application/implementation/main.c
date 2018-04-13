#include <stdio.h>

#include "settings.h"
#include "trainsession.h"
#include "testsession.h"
#include "layer_commons.h"
#include "network.h"
#include "weightgenerator.h"

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
        cog.outl(".relevant_columns_offset = " + str(current._weight_off))
        cog.outl("},")
        cog.outl(".weight_shape = ")
        cog.outl("{")
        cog.outl(".relevant_entries_count = " + str(current.get_weight_shape().get_count_output()) + ",")
        cog.outl(".relevant_entries_offset = " + str(current._act_mem_i_off))
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
.input_activation_count = 7840,
.output_activation_offset = 0,
.output_activation_count = 92700,
.filter_feature_input_count = 1,
.filter_feature_input_count = 1,
.filter_x_count = 5,
.filter_y_count = 5,
.filter_feature_output_count = 12,
.batch_count = 10,
.filter_matrix_width = 25,
.input_matrix_height = 5,
.input_matrix_width = 1545,
.input_matrix_toplayer_elements_count = 28,
.partial_output_matrix_count = 18540,
.full_output_matrix_width = 7725,
.weights_total_count = 312,
.weights_offset = 0,
.biases_offset = 300,
},
.layer_1 = 
{
// maxpooling
.input_activation_offset = 0,
.input_activation_count = 92700,
.output_activation_offset = 92704,
.output_activation_count = 17280,
.pooling_layout = 
{
.relevant_entries_count = 69120,
.num_of_lines = 17280,
.relevant_columns_per_line = 4,
.relevant_columns_offset = 0
},
.weight_shape = 
{
.relevant_entries_count = 17280,
.relevant_entries_offset = 0
}
},
.layer_2 = 
{
// convolutional
.input_activation_offset = 92704,
.input_activation_count = 17280,
.output_activation_offset = 109984,
.output_activation_count = 22240,
.filter_feature_input_count = 12,
.filter_feature_input_count = 12,
.filter_x_count = 5,
.filter_y_count = 5,
.filter_feature_output_count = 16,
.batch_count = 10,
.filter_matrix_width = 300,
.input_matrix_height = 60,
.input_matrix_width = 278,
.input_matrix_toplayer_elements_count = 144,
.partial_output_matrix_count = 4448,
.full_output_matrix_width = 1390,
.weights_total_count = 4816,
.weights_offset = 320,
.biases_offset = 5120,
},
.layer_3 = 
{
// maxpooling
.input_activation_offset = 109984,
.input_activation_count = 22240,
.output_activation_offset = 132224,
.output_activation_count = 2560,
.pooling_layout = 
{
.relevant_entries_count = 10240,
.num_of_lines = 2560,
.relevant_columns_per_line = 4,
.relevant_columns_offset = 69120
},
.weight_shape = 
{
.relevant_entries_count = 2560,
.relevant_entries_offset = 17280
}
},
.layer_4 = 
{
// fully connected
.input_activation_offset = 132224,
.input_activation_count = 2560,
.output_activation_offset = 134784,
.output_activation_count = 100,
.single_input_count = 256,
.single_output_count = 10,
.batch_count = 10,
.weights_count_total = 2570,
.weights_offset = 5136,
.biases_offset = 7696,
}
};
// [[[end]]]


DataSupplier_t trainsupplier, testsupplier;
NetState_t netstate;


int main(void)
{
    Int_t iteration;
    Float_t test_accuracy;
    // allocate memory nor network execution
    netstate_init(&netstate);
    // generate weights for the network
    weightgen_generate(NETWORK_WEIGHTS_F_SIZE, netstate.weights_f);
    // initialize suppliers to read input data and labels
    datasupply_init(&trainsupplier, CONFIG_NUM_TRAINFILES, CONFIG_DIR_TRAIN);
    datasupply_init(&testsupplier, CONFIG_NUM_TESTFILES, CONFIG_DIR_TEST);
    // test with random weights first; just for later comparison
    test_accuracy = exec_testsession(&network, &netstate, &testsupplier, CONFIG_TESTS_PER_ITERATION);
    printf("Initial accuracy: ");
    printf(FLOAT_T_ESCAPE, test_accuracy);
    printf("\n");
    // do iterations consisting of training and testing
    for(iteration=1; iteration<=CONFIG_NUM_OF_ITERATIONS; iteration++)
    {
        exec_trainsession(&network, &netstate, &trainsupplier, CONFIG_TRAININGS_PER_ITERATION);
        test_accuracy = exec_testsession(&network, &netstate, &testsupplier, CONFIG_TESTS_PER_ITERATION);
        printf("Iteration %d: ", iteration);
        printf(FLOAT_T_ESCAPE, test_accuracy);
        printf("\n");
    }
    return 0;
}
