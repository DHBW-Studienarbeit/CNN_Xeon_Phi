#include "network.h"
#include "mathematics.h"
#include "shared_arrays.h"


Float_t net_activations[NETWORK_ACTIVATION_SIZE];
Float_t net_activations_errors[NETWORK_ACTIVATION_SIZE];
Int_t net_pooling_mem[NETWORK_POOLING_MEM_SIZE];
Float_t net_weights_f[NETWORK_WEIGHTS_F_SIZE];
Float_t net_weights_f_errors[NETWORK_WEIGHTS_F_SIZE];
Int_t net_weights_i[NETWORK_WEIGHTS_I_SIZE];


void network_forward(NeuronalNetwork_p network, Float_p input)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net

    i=0
    first = True
    for current in net._layers:
        if current.__class__.__name__ == "MaxPoolingLayer":
            cog.out(current.get_C_forwardname(first) + "(network->layer_" \
                    + str(i) + ", net_activations, net_pooling_mem")
            if first==True:
                cog.out(", input")
            cog.outl(");")
        else:
            cog.out(current.get_C_forwardname(first) + "(network->layer_" \
                    + str(i) + ", net_activations")
            if first==True:
                cog.out(", input")
            cog.outl(");")
        i=i+1
        first = False
    ]]] */
    layer_conv_first_forward(network->layer_0, net_activations, input);
    layer_maxpool_forward(network->layer_1, net_activations, net_pooling_mem);
    layer_conv_forward(network->layer_2, net_activations);
    layer_maxpool_forward(network->layer_3, net_activations, net_pooling_mem);
    layer_fullcon_forward(network->layer_4, net_activations);
    // [[[end]]]
}

void network_backward(NeuronalNetwork_p network, Float_p input)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net

    i=len(net._layers)
    first = False
    for current in reversed(net._layers):
        i = i - 1
        if i == 0:
            first = True
        if current.__class__.__name__ == "MaxPoolingLayer":
            cog.out(current.get_C_backwardname(first) + "(network->layer_" \
                    + str(i) + ", net_activations, net_pooling_mem")
            if first==True:
                cog.out(", input")
            cog.outl(", net_activations_errors, net_weights_f_errors);")
        else:
            cog.out(current.get_C_backwardname(first) + "(network->layer_" \
                    + str(i) + ", net_activations")
            if first==True:
                cog.out(", input")
            cog.outl(", net_activations_errors, net_weights_f_errors);")
    ]]] */
    layer_fullcon_backward(network->layer_4, net_activations, net_activations_errors, net_weights_f_errors);
    layer_maxpool_backward(network->layer_3, net_activations, net_pooling_mem, net_activations_errors, net_weights_f_errors);
    layer_conv_backward(network->layer_2, net_activations, net_activations_errors, net_weights_f_errors);
    layer_maxpool_backward(network->layer_1, net_activations, net_pooling_mem, net_activations_errors, net_weights_f_errors);
    layer_conv_first_backward(network->layer_0, net_activations, input, net_activations_errors, net_weights_f_errors);
    // [[[end]]]
}


void network_gradient_descent(NeuronalNetwork_p network, Float_t learn_rate)
{
    MATH_VECT_VECT_SCAL_ADD(    NETWORK_WEIGHTS_F_SIZE,
                                (-1.0f)*learn_rate,
                                net_weights_f_errors,
                                1,
                                net_weights_f,
                                1 );
}

Float_p network_get_cost(NeuronalNetwork_p network, Float_p labels)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net
    cog.out("get_cost(")
    cog.out(str(net.get_output_shape().get_count_probes()))
    cog.out(", ")
    cog.out(str(net.get_output_shape().get_count_total()//net.get_output_shape().get_count_probes()))
    cog.out(", net_activations + network->label_")
    cog.out(str(len(net._layers)-1))
    cog.out(".output_activation_offset, labels, shared_tmp_floats);")
    ]]] */
    get_cost(10, 10, net_activations + network->label_4.output_activation_offset, labels, shared_tmp_floats);
    // [[[end]]]
}

void network_derive_cost(NeuronalNetwork_p network, Float_p labels)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net
    cog.out("get_cost(")
    cog.out(str(net.get_output_shape().get_count_total()))
    cog.out(", net_activations + network->label_")
    cog.out(str(len(net._layers)-1))
    cog.out(".output_activation_offset, labels, net_activations_errors + network->label_")
    cog.out(str(len(net._layers)-1))
    cog.out(".output_activation_offset);")
    ]]] */
    get_cost(100, net_activations + network->label_4.output_activation_offset, labels, net_activations_errors + network->label_4.output_activation_offset);
    // [[[end]]]
}
