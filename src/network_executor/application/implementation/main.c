#include <stdio.h>

#include "settings.h"
#include "trainsession.h"
#include "testsession.h"
#include "network.h"
#include "weightgenerator.h"
#include "testing.h"

#include "mathematics.h"
#include "shared_arrays.h"

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
        cog.outl(".input_x_count = " + str(current.get_input_shape().get_count_x()) + ",")
        cog.outl(".input_xy_count = " + str(current.get_input_shape().get_count_x()*current.get_input_shape().get_count_y()) + ",")
        cog.outl(".weights_total_count = " + str(current.get_weight_shape()._count_total) + ",")
        cog.outl(".weights_offset = " + str(current._weight_off) + ",")
        cog.outl(".biases_offset = " + str(current._bias_off) + ",")
    if current.__class__.__name__ == "MaxPoolingLayer":
        cog.outl("// maxpooling")
        cog.outl(".input_activation_offset = " + str(current._act_in_off) + ",")
        cog.outl(".input_activation_count = " + str(current.get_input_shape().get_count_total()) + ",")
        cog.outl(".output_activation_offset = " + str(current._act_out_off) + ",")
        cog.outl(".output_activation_count = " + str(current.get_output_shape().get_count_total()) + ",")

        cog.outl(".output_p = " + str(current.get_output_shape().get_count_probes()) + ",")
        cog.outl(".output_y = " + str(current.get_output_shape().get_count_y()) + ",")
        cog.outl(".output_x = " + str(current.get_output_shape().get_count_x()) + ",")
        cog.outl(".output_f = " + str(current.get_output_shape().get_count_features()) + ",")

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
.input_activation_count = 6272,
.output_activation_offset = 0,
.output_activation_count = 197120,
.filter_feature_input_count = 1,
.filter_x_count = 5,
.filter_y_count = 5,
.filter_feature_output_count = 32,
.batch_count = 8,
.filter_matrix_width = 25,
.input_matrix_height = 5,
.input_matrix_width = 1232,
.input_matrix_toplayer_elements_count = 28,
.partial_output_matrix_count = 39424,
.full_output_matrix_width = 6160,
.input_x_count = 28,
.input_xy_count = 784,
.weights_total_count = 832,
.weights_offset = 0,
.biases_offset = 800,
},
.layer_1 = 
{
// maxpooling
.input_activation_offset = 0,
.input_activation_count = 197120,
.output_activation_offset = 197120,
.output_activation_count = 36864,
.output_p = 8,
.output_y = 12,
.output_x = 12,
.output_f = 32,
.pooling_layout = 
{
.relevant_entries_count = 147456,
.num_of_lines = 36864,
.relevant_columns_per_line = 4,
.relevant_columns_offset = 0
},
.weight_shape = 
{
.relevant_entries_count = 36864,
.relevant_entries_offset = 0
}
},
.layer_2 = 
{
// convolutional
.input_activation_offset = 197120,
.input_activation_count = 36864,
.output_activation_offset = 233984,
.output_activation_count = 70400,
.filter_feature_input_count = 32,
.filter_x_count = 5,
.filter_y_count = 5,
.filter_feature_output_count = 64,
.batch_count = 8,
.filter_matrix_width = 800,
.input_matrix_height = 160,
.input_matrix_width = 220,
.input_matrix_toplayer_elements_count = 384,
.partial_output_matrix_count = 14080,
.full_output_matrix_width = 1100,
.input_x_count = 12,
.input_xy_count = 144,
.weights_total_count = 51264,
.weights_offset = 832,
.biases_offset = 52032,
},
.layer_3 = 
{
// maxpooling
.input_activation_offset = 233984,
.input_activation_count = 70400,
.output_activation_offset = 304384,
.output_activation_count = 8192,
.output_p = 8,
.output_y = 4,
.output_x = 4,
.output_f = 64,
.pooling_layout = 
{
.relevant_entries_count = 32768,
.num_of_lines = 8192,
.relevant_columns_per_line = 4,
.relevant_columns_offset = 147456
},
.weight_shape = 
{
.relevant_entries_count = 8192,
.relevant_entries_offset = 36864
}
},
.layer_4 = 
{
// fully connected
.input_activation_offset = 304384,
.input_activation_count = 8192,
.output_activation_offset = 312576,
.output_activation_count = 8192,
.single_input_count = 1024,
.single_output_count = 1024,
.batch_count = 8,
.weights_count_total = 1049600,
.weights_offset = 52096,
.biases_offset = 1100672,
},
.layer_5 = 
{
// fully connected
.input_activation_offset = 312576,
.input_activation_count = 8192,
.output_activation_offset = 320768,
.output_activation_count = 80,
.single_input_count = 1024,
.single_output_count = 10,
.batch_count = 8,
.weights_count_total = 10250,
.weights_offset = 1101696,
.biases_offset = 1111936,
}
};
// [[[end]]]


DataSupplier_t trainsupplier, testsupplier;
NetState_t netstate;


int main(void)
{
    TestResult_t testresult;
    Int_t iteration;
    Float_t last_cost;
    Float_t learning_rate = CONFIG_LEARNING_RATE;
    printf("do some initialization stuff\n");
    init_shared_arrays();
    // allocate memory for network execution
    netstate_init(&network, &netstate);
    // generate weights for the network
    weightgen_generate(NETWORK_WEIGHTS_F_SIZE, netstate.weights_f);
    // initialize suppliers to read input data and labels
    datasupply_init(&trainsupplier, CONFIG_NUM_TRAINFILES, CONFIG_DIR_TRAIN);
    datasupply_init(&testsupplier, CONFIG_NUM_TESTFILES, CONFIG_DIR_TEST);
    // test with random weights first; just for later comparison
    exec_testsession(&network, &netstate, &testsupplier, CONFIG_TESTS_PER_ITERATION, &testresult);
    printf("Iteration 0: ");
    printf(FLOAT_T_ESCAPE, testresult.accuracy);
    printf("\t");
    printf(FLOAT_T_ESCAPE, testresult.cost);
    printf("\n");
    last_cost = testresult.cost;
    // do iterations consisting of training and testing
    for(iteration=1; iteration<=CONFIG_NUM_OF_ITERATIONS; iteration++)
    {
        exec_trainsession(&network, &netstate, &trainsupplier, CONFIG_TRAININGS_PER_ITERATION, learning_rate);
        exec_testsession(&network, &netstate, &testsupplier, CONFIG_TESTS_PER_ITERATION, &testresult);
        printf("Iteration %d: ", iteration);
        printf(FLOAT_T_ESCAPE, testresult.accuracy);
        printf("\t");
        printf(FLOAT_T_ESCAPE, testresult.cost);
        printf("\n");
#ifdef CONFIG_LEARNRATE_REDUCTION
        if(testresult.cost > last_cost)
        {
            learning_rate *= CONFIG_LEARNRATE_REDUCTION;
            printf("Setting learning rate to ");
            printf(FLOAT_T_ESCAPE, learning_rate);
            printf("\n");
        }
        last_cost = testresult.cost;
#endif
    }
    return 0;
}
