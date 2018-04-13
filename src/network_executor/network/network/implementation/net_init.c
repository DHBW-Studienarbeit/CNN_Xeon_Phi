#include "settings.h"
#include "netstate.h"
#include "network.h"
#include "mathematics.h"

extern const Int_t net_weights_i[NETWORK_WEIGHTS_I_SIZE];

void netstate_init(const NeuronalNetwork_p network, NetState_p netstate)
{
    Int_t i;
    MaxPoolingLayer_p current_layer;
    netstate->activations = MATH_MALLOC_F_ARRAY(NETWORK_ACTIVATION_SIZE);
    netstate->activations_errors = MATH_MALLOC_F_ARRAY(NETWORK_ACTIVATION_SIZE);
    netstate->weights_f = MATH_MALLOC_F_ARRAY(NETWORK_WEIGHTS_F_SIZE);
    netstate->weights_f_errors = MATH_MALLOC_F_ARRAY(NETWORK_WEIGHTS_F_SIZE);
    netstate->weights_i = MATH_MALLOC_I_ARRAY(NETWORK_WEIGHTS_I_SIZE);
    netstate->pooling_mem = MATH_MALLOC_I_ARRAY(NETWORK_POOLING_MEM_SIZE);

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
    for(i=0; i<17280; i++)
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
    for(i=0; i<2560; i++)
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
    netstate->weights_i[69120 + i*filter_size+j] = 
    layer_conv_get_output_position(&(network->layer_2), netstate, p_out, y_in, x_in, f_out);
    j++;
    }
    }
    }
    // [[[end]]]

}
