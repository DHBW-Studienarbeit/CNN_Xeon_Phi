#include "settings.h"
#include "netstate.h"
#include "network.h"
#include "mathematics.h"

extern const Int_t net_weights_i[NETWORK_WEIGHTS_I_SIZE];

void network_init(const NeuronalNetwork_p network)
{
    Int_t i;
    MaxPoolingLayer_p current_layer;

    network->activations = MATH_MALLOC_F_ARRAY(NETWORK_ACTIVATION_SIZE);
    network->activations_errors = MATH_MALLOC_F_ARRAY(NETWORK_ACTIVATION_SIZE);
    network->weights_f = MATH_MALLOC_F_ARRAY(NETWORK_WEIGHTS_F_SIZE);
    network->weights_f_errors = MATH_MALLOC_F_ARRAY(NETWORK_WEIGHTS_F_SIZE);
    network->weights_i = MATH_MALLOC_I_ARRAY(NETWORK_WEIGHTS_I_SIZE);
    network->pooling_mem = MATH_MALLOC_I_ARRAY(NETWORK_POOLING_MEM_SIZE);

    network->shared_ones_floats = MATH_MALLOC_F_ARRAY(SHARED_ARRAY_SIZE);
    network->shared_tmp_floats = MATH_MALLOC_F_ARRAY(SHARED_TMP_ARRAY_SIZE);
    network->shared_tmp_ints = MATH_MALLOC_I_ARRAY(SHARED_TMP_ARRAY_SIZE);

    for(i=0; i<SHARED_ARRAY_SIZE; i++)
    {
        shared_ones_floats[i] = 1.0f;
    }


    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net
    i=0
    for current in net._layers:
        prefix = "network->layer_" + str(i)
        if current.__class__.__name__ == "FullyConnectedLayer":
            cog.outl("// fully connected")
            cog.outl(prefix + ".input_activation_start = network->activations + " + str(current._act_in_off) + ";")
            cog.outl(prefix + ".input_activation_error_start = network->activations_errors + " + str(current._act_in_off) + ";")
            cog.outl(prefix + ".input_activation_count = " + str(current.get_input_shape().get_count_total()) + ";")
            cog.outl(prefix + ".output_activation_start = network->activations + " + str(current._act_out_off) + ";")
            cog.outl(prefix + ".output_activation_error_start = network->activations_errors + " + str(current._act_out_off) + ";")
            cog.outl(prefix + ".output_activation_count = " + str(current.get_output_shape().get_count_total()) + ";")
            cog.outl(prefix + ".single_input_count = " + str(current.get_weight_shape().get_count_input()) + ";")
            cog.outl(prefix + ".single_output_count = " + str(current.get_weight_shape().get_count_output()) + ";")
            cog.outl(prefix + ".batch_count = " + str(current.get_input_shape().get_count_probes()) + ";")
            cog.outl(prefix + ".weights_count_total = " + str(current.get_weight_shape().get_count_total()) + ";")
            cog.outl(prefix + ".weights_start = network->weights_f + " + str(current._weight_off) + ";")
            cog.outl(prefix + ".weights_error_start = network->weights_f_errors + " + str(current._weight_off) + ";")
            cog.outl(prefix + ".biases_start = network->weights_f + " + str(current._bias_off) + ";")
            cog.outl(prefix + ".biases_error_start = network->weights_f_errors + " + str(current._bias_off) + ";")
            cog.outl(prefix + ".shared_tmp_floats = network->shared_tmp_floats;")
            cog.outl(prefix + ".shared_ones_floats = network->shared_ones_floats;")
        if current.__class__.__name__ == "ConvolutionalLayer":
            cog.outl("// convolutional")
            cog.outl(prefix + ".input_activation_start = network->activations + " + str(current._act_in_off) + ";")
            cog.outl(prefix + ".input_activation_error_start = network->activations_errors + " + str(current._act_in_off) + ";")
            cog.outl(prefix + ".input_activation_count = " + str(current.get_input_shape().get_count_total()) + ";")
            cog.outl(prefix + ".output_activation_start = network->activations + " + str(current._act_out_off) + ";")
            cog.outl(prefix + ".output_activation_error_start = network->activations_errors + " + str(current._act_out_off) + ";")
            cog.outl(prefix + ".output_activation_count = " + str(current.get_output_shape().get_count_total()) + ";")
            cog.outl(prefix + ".filter_feature_input_count = " + str(current.get_input_shape().get_count_features()) + ";")
            cog.outl(prefix + ".filter_x_count = " + str(current.get_output_shape()._filter_size_x) + ";")
            cog.outl(prefix + ".filter_y_count = " + str(current.get_output_shape()._filter_size_y) + ";")
            cog.outl(prefix + ".filter_feature_output_count = " + str(current.get_output_shape().get_count_features()) + ";")
            cog.outl(prefix + ".batch_count = " + str(current.get_input_shape().get_count_probes()) + ";")
            cog.outl(prefix + ".filter_matrix_width = " + str(current.get_weight_shape().get_count_input()) + ";")
            cog.outl(prefix + ".input_matrix_height = " + str(current.get_input_shape().get_count_features() * current.get_output_shape()._filter_size_x) + ";")
            cog.outl(prefix + ".input_matrix_width = " + str(current.get_output_shape()._num_output_section_columns) + ";")
            cog.outl(prefix + ".input_matrix_toplayer_elements_count = " + str(current.get_input_shape().get_count_x()*current.get_input_shape().get_count_features()) + ";")
            cog.outl(prefix + ".partial_output_matrix_count = " + str(current.get_output_shape()._output_section_size) + ";")
            cog.outl(prefix + ".full_output_matrix_width = " + str(current.get_output_shape()._num_output_section_columns * current.get_output_shape()._filter_size_y) + ";")
            cog.outl(prefix + ".input_x_count = " + str(current.get_input_shape().get_count_x()) + ";")
            cog.outl(prefix + ".input_xy_count = " + str(current.get_input_shape().get_count_x()*current.get_input_shape().get_count_y()) + ";")
            cog.outl(prefix + ".weights_total_count = " + str(current.get_weight_shape()._count_total) + ";")
            cog.outl(prefix + ".weights_start = network->weights_f + " + str(current._weight_off) + ";")
            cog.outl(prefix + ".weights_error_start = network->weights_f_errors + " + str(current._weight_off) + ";")
            cog.outl(prefix + ".biases_start = network->weights_f + " + str(current._bias_off) + ";")
            cog.outl(prefix + ".biases_error_start = network->weights_f_errors + " + str(current._bias_off) + ";")        
            cog.outl(prefix + ".shared_tmp_floats = network->shared_tmp_floats;")
            cog.outl(prefix + ".shared_ones_floats = network->shared_ones_floats;")
        if current.__class__.__name__ == "MaxPoolingLayer":
            cog.outl("// maxpooling")
            cog.outl(prefix + ".input_activation_offset = " + str(current._act_in_off) + ";")
            cog.outl(prefix + ".input_activation_count = " + str(current.get_input_shape().get_count_total()) + ";")
            cog.outl(prefix + ".output_activation_offset = " + str(current._act_out_off) + ";")
            cog.outl(prefix + ".output_activation_count = " + str(current.get_output_shape().get_count_total()) + ";")
            cog.outl(prefix + ".output_p = " + str(current.get_output_shape().get_count_probes()) + ";")
            cog.outl(prefix + ".output_y = " + str(current.get_output_shape().get_count_y()) + ";")
            cog.outl(prefix + ".output_x = " + str(current.get_output_shape().get_count_x()) + ";")
            cog.outl(prefix + ".output_f = " + str(current.get_output_shape().get_count_features()) + ";")
            cog.outl(prefix + ".pooling_layout.relevant_entries_count = " + str(current.get_weight_shape().get_count_total()) + ";")
            cog.outl(prefix + ".pooling_layout.num_of_lines = " + str(current.get_weight_shape().get_count_output()) + ";")
            cog.outl(prefix + ".pooling_layout.relevant_columns_per_line = " + str(current._filter_size_x * current._filter_size_y) + ";")
            cog.outl(prefix + ".pooling_layout.relevant_columns_offset = " + str(current._weight_off) + ";")
            cog.outl(prefix + ".weight_shape.relevant_entries_count = " + str(current.get_weight_shape().get_count_output()) + ";")
            cog.outl(prefix + ".weight_shape.relevant_entries_offset = " + str(current._act_mem_i_off) + ";")
        i = i + 1
    ]]] */
    // convolutional
    network->layer_0.input_activation_offset = 0;
    network->layer_0.input_activation_count = 6272;
    network->layer_0.output_activation_offset = 0;
    network->layer_0.output_activation_count = 197120;
    network->layer_0.filter_feature_input_count = 1;
    network->layer_0.filter_x_count = 5;
    network->layer_0.filter_y_count = 5;
    network->layer_0.filter_feature_output_count = 32;
    network->layer_0.batch_count = 8;
    network->layer_0.filter_matrix_width = 25;
    network->layer_0.input_matrix_height = 5;
    network->layer_0.input_matrix_width = 1232;
    network->layer_0.input_matrix_toplayer_elements_count = 28;
    network->layer_0.partial_output_matrix_count = 39424;
    network->layer_0.full_output_matrix_width = 6160;
    network->layer_0.input_x_count = 28;
    network->layer_0.input_xy_count = 784;
    network->layer_0.weights_total_count = 832;
    network->layer_0.weights_offset = 0;
    network->layer_0.biases_offset = 800;
    // maxpooling
    network->layer_1.input_activation_offset = 0;
    network->layer_1.input_activation_count = 197120;
    network->layer_1.output_activation_offset = 197120;
    network->layer_1.output_activation_count = 36864;
    network->layer_1.output_p = 8;
    network->layer_1.output_y = 12;
    network->layer_1.output_x = 12;
    network->layer_1.output_f = 32;
    network->layer_1.pooling_layout.relevant_entries_count = 147456;
    network->layer_1.pooling_layout.num_of_lines = 36864;
    network->layer_1.pooling_layout.relevant_columns_per_line = 4;
    network->layer_1.pooling_layout.relevant_columns_offset = 0;
    network->layer_1.weight_shape.relevant_entries_count = 36864;
    network->layer_1.weight_shape.relevant_entries_offset = 0;
    // convolutional
    network->layer_2.input_activation_offset = 197120;
    network->layer_2.input_activation_count = 36864;
    network->layer_2.output_activation_offset = 233984;
    network->layer_2.output_activation_count = 70400;
    network->layer_2.filter_feature_input_count = 32;
    network->layer_2.filter_x_count = 5;
    network->layer_2.filter_y_count = 5;
    network->layer_2.filter_feature_output_count = 64;
    network->layer_2.batch_count = 8;
    network->layer_2.filter_matrix_width = 800;
    network->layer_2.input_matrix_height = 160;
    network->layer_2.input_matrix_width = 220;
    network->layer_2.input_matrix_toplayer_elements_count = 384;
    network->layer_2.partial_output_matrix_count = 14080;
    network->layer_2.full_output_matrix_width = 1100;
    network->layer_2.input_x_count = 12;
    network->layer_2.input_xy_count = 144;
    network->layer_2.weights_total_count = 51264;
    network->layer_2.weights_offset = 832;
    network->layer_2.biases_offset = 52032;
    // maxpooling
    network->layer_3.input_activation_offset = 233984;
    network->layer_3.input_activation_count = 70400;
    network->layer_3.output_activation_offset = 304384;
    network->layer_3.output_activation_count = 8192;
    network->layer_3.output_p = 8;
    network->layer_3.output_y = 4;
    network->layer_3.output_x = 4;
    network->layer_3.output_f = 64;
    network->layer_3.pooling_layout.relevant_entries_count = 32768;
    network->layer_3.pooling_layout.num_of_lines = 8192;
    network->layer_3.pooling_layout.relevant_columns_per_line = 4;
    network->layer_3.pooling_layout.relevant_columns_offset = 147456;
    network->layer_3.weight_shape.relevant_entries_count = 8192;
    network->layer_3.weight_shape.relevant_entries_offset = 36864;
    // fully connected
    network->layer_4.input_activation_start = network->activations + 304384;
    network->layer_4.input_activation_error_start = network->activations_errors + 304384;
    network->layer_4.input_activation_count = 8192;
    network->layer_4.output_activation_start = network->activations + 312576;
    network->layer_4.output_activation_error_start = network->activations_errors + 312576;
    network->layer_4.output_activation_count = 8192;
    network->layer_4.single_input_count = 1024;
    network->layer_4.single_output_count = 1024;
    network->layer_4.batch_count = 8;
    network->layer_4.weights_count_total = 1049600;
    network->layer_4.weights_start = network->weights_f + 52096;
    network->layer_4.weights_error_start = network->weights_f_errors + 52096;
    network->layer_4.biases_start = network->weights_f + 1100672;
    network->layer_4.biases_error_start = network->weights_f_errors + 1100672;
    // fully connected
    network->layer_5.input_activation_start = network->activations + 312576;
    network->layer_5.input_activation_error_start = network->activations_errors + 312576;
    network->layer_5.input_activation_count = 8192;
    network->layer_5.output_activation_start = network->activations + 320768;
    network->layer_5.output_activation_error_start = network->activations_errors + 320768;
    network->layer_5.output_activation_count = 80;
    network->layer_5.single_input_count = 1024;
    network->layer_5.single_output_count = 10;
    network->layer_5.batch_count = 8;
    network->layer_5.weights_count_total = 10250;
    network->layer_5.weights_start = network->weights_f + 1101696;
    network->layer_5.weights_error_start = network->weights_f_errors + 1101696;
    network->layer_5.biases_start = network->weights_f + 1111936;
    network->layer_5.biases_error_start = network->weights_f_errors + 1111936;
    // [[[end]]]

    // prepare weights for maxpooling
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net
    i=0
    for current in net._layers:
        if current.__class__.__name__ == "MaxPoolingLayer":
            previous = net._layers[i-1]
            cog.outl("// layer_" + str(i))
            cog.outl("current_layer = &(network->layer_" + str(i) + ");")
            cog.outl("#pragma omp parallel for")
            cog.outl("for(i=0; i<" + str(current.get_output_shape().get_count_total()) + "; i++)")
            cog.outl("{")
            cog.outl("Int_t f_out = i % current_layer->output_f;")
            cog.outl("Int_t inner = i / current_layer->output_f;")
            cog.outl("Int_t x_out = inner % current_layer->output_x;")
            cog.outl("inner = inner / current_layer->output_x;")
            cog.outl("Int_t y_out = inner % current_layer->output_y;")
            cog.outl("inner = inner / current_layer->output_y;")
            cog.outl("Int_t p_out = inner % current_layer->output_p;")
            cog.outl("inner = inner / current_layer->output_p;")
            cog.outl("Int_t y_filter, x_filter, y_in, x_in;")
            cog.outl("Int_t filter_size = " + str(current._filter_size_y * current._filter_size_x) + ";")
            cog.outl("Int_t j=0;")
            cog.outl("for(y_filter=0; y_filter<" + str(current._filter_size_y) + "; y_filter++)")
            cog.outl("{")
            cog.outl("y_in = " + str(current._filter_size_y) + " * y_out + y_filter;")
            cog.outl("for(x_filter=0; x_filter<" + str(current._filter_size_x) + "; x_filter++)")
            cog.outl("{")
            cog.outl("x_in = " + str(current._filter_size_x) + " * x_out + x_filter;")
            cog.outl("netstate->weights_i[" + str(current._weight_off) + " + i*filter_size+j] = ")
            cog.outl(previous.get_C_outputname() + "_position(&(network->layer_" + str(i-1) + "), netstate, p_out, y_in, x_in, f_out);")
            cog.outl("j++;")
            cog.outl("}")
            cog.outl("}")
            cog.outl("}")
        i=i+1
    ]]] */
    // layer_1
    current_layer = &(network->layer_1);
    #pragma omp parallel for
    for(i=0; i<36864; i++)
    {
    Int_t f_out = i % current_layer->output_f;
    Int_t inner = i / current_layer->output_f;
    Int_t x_out = inner % current_layer->output_x;
    inner = inner / current_layer->output_x;
    Int_t y_out = inner % current_layer->output_y;
    inner = inner / current_layer->output_y;
    Int_t p_out = inner % current_layer->output_p;
    inner = inner / current_layer->output_p;
    Int_t y_filter, x_filter, y_in, x_in;
    Int_t filter_size = 4;
    Int_t j=0;
    for(y_filter=0; y_filter<2; y_filter++)
    {
    y_in = 2 * y_out + y_filter;
    for(x_filter=0; x_filter<2; x_filter++)
    {
    x_in = 2 * x_out + x_filter;
    netstate->weights_i[0 + i*filter_size+j] =
    layer_conv_get_output_position(&(network->layer_0), netstate, p_out, y_in, x_in, f_out);
    j++;
    }
    }
    }
    // layer_3
    current_layer = &(network->layer_3);
    #pragma omp parallel for
    for(i=0; i<8192; i++)
    {
    Int_t f_out = i % current_layer->output_f;
    Int_t inner = i / current_layer->output_f;
    Int_t x_out = inner % current_layer->output_x;
    inner = inner / current_layer->output_x;
    Int_t y_out = inner % current_layer->output_y;
    inner = inner / current_layer->output_y;
    Int_t p_out = inner % current_layer->output_p;
    inner = inner / current_layer->output_p;
    Int_t y_filter, x_filter, y_in, x_in;
    Int_t filter_size = 4;
    Int_t j=0;
    for(y_filter=0; y_filter<2; y_filter++)
    {
    y_in = 2 * y_out + y_filter;
    for(x_filter=0; x_filter<2; x_filter++)
    {
    x_in = 2 * x_out + x_filter;
    netstate->weights_i[147456 + i*filter_size+j] =
    layer_conv_get_output_position(&(network->layer_2), netstate, p_out, y_in, x_in, f_out);
    j++;
    }
    }
    }
    // [[[end]]]

}
