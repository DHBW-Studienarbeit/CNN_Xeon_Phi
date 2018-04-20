#include "network.h"
#include "mathematics.h"


void network_forward(const NeuronalNetwork_p network, const Float_p input)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net

    i=0
    first = True
    for current in net._layers:
        cog.out(current.get_C_forwardname(first) + "(&(network->layer_" \
                + str(i) + ")")
        if first==True:
            cog.out(", input")
        cog.outl(");")
        i=i+1
        first = False
    ]]] */
    layer_conv_first_forward(&(network->layer_0), input);
    layer_maxpool_forward(&(network->layer_1));
    layer_conv_forward(&(network->layer_2));
    layer_maxpool_forward(&(network->layer_3));
    layer_fullcon_forward(&(network->layer_4));
    layer_fullcon_forward(&(network->layer_5));
    // [[[end]]]
}

void network_backward(const NeuronalNetwork_p network, const Float_p input)
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
        cog.out(current.get_C_backwardname(first) + "(&(network->layer_" \
                + str(i) + ")")
        if first==True:
            cog.out(", input")
        cog.outl(");")
    ]]] */
    layer_fullcon_backward(&(network->layer_5));
    layer_fullcon_backward(&(network->layer_4));
    layer_maxpool_backward(&(network->layer_3));
    layer_conv_backward(&(network->layer_2));
    layer_maxpool_backward(&(network->layer_1));
    layer_conv_first_backward(&(network->layer_0), input);
    // [[[end]]]
}


void network_gradient_descent(const NeuronalNetwork_p network, Float_t learn_rate)
{
    MATH_VECT_VECT_SCAL_ADD(    NETWORK_WEIGHTS_F_SIZE,
                                (-1.0f)*learn_rate,
                                network->weights_f_errors,
                                1,
                                network->weights_f,
                                1 );
}

Float_t network_get_cost(const NeuronalNetwork_p network, const Float_p labels)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net
    cog.out("return get_cost(")
    cog.out(str(net.get_output_shape().get_count_probes()))
    cog.out(", ")
    cog.out(str(net.get_output_shape().get_count_total()//net.get_output_shape().get_count_probes()))
    cog.out(", network->layer_")
    cog.out(str(len(net._layers)-1))
    cog.out(".output_activation_start, labels, network->shared_tmp_floats);")
    ]]] */
    return get_cost(16, 10, network->layer_5.output_activation_start, labels, network->shared_tmp_floats);
    // [[[end]]]
}


Float_t network_get_accuracy(const NeuronalNetwork_p network, const Float_p labels)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net
    cog.out("return get_accuracy(")
    cog.out(str(net.get_output_shape().get_count_probes()))
    cog.out(", ")
    cog.out(str(net.get_output_shape().get_count_total()//net.get_output_shape().get_count_probes()))
    cog.out(", network->layer_")
    cog.out(str(len(net._layers)-1))
    cog.out(".output_activation_start, labels);")
    ]]] */
    return get_accuracy(16, 10, network->layer_5.output_activation_start, labels);
    // [[[end]]]
}


void network_derive_cost(const NeuronalNetwork_p network, const Float_p labels)
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net
    cog.out("get_cost_derivatives(")
    cog.out(str(net.get_output_shape().get_count_probes()))
    cog.out(", ")
    cog.out(str(net.get_output_shape().get_count_total()//net.get_output_shape().get_count_probes()))
    cog.out(", network->layer_")
    cog.out(str(len(net._layers)-1))
    cog.out(".output_activation_start, labels, network->layer_")
    cog.out(str(len(net._layers)-1))
    cog.out(".output_activation_error_start, network->shared_tmp_floats);")
    ]]] */
    get_cost_derivatives(16, 10, network->layer_5.output_activation_start, labels, network->layer_5.output_activation_error_start, network->shared_tmp_floats);
    // [[[end]]]
}
