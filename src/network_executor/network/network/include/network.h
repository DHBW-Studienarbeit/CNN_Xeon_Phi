#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "settings.h"
#include "netstate.h"
#include "fullyconnected_layer.h"
#include "convlayer.h"
#include "maxpoollayer.h"


typedef struct
{
    /* [[[cog
    import cog
    from network_descriptor.NetInstance import net

    i=0
    for current in net._layers:
        cog.outl(current.get_C_typename() + "t layer_" + str(i) + ";")
        i=i+1
    ]]] */
    ConvolutionalLayer_t layer_0;
    MaxPoolingLayer_t layer_1;
    ConvolutionalLayer_t layer_2;
    MaxPoolingLayer_t layer_3;
    FullyConnectedLayer_t layer_4;
    FullyConnectedLayer_t layer_5;
    // [[[end]]]
} NeuronalNetwork_t, *NeuronalNetwork_p;


void netstate_init(const NeuronalNetwork_p network, NetState_p netstate);

void network_forward(const NeuronalNetwork_p network, NetState_p netstate, const Float_p input);
void network_backward(const NeuronalNetwork_p network, NetState_p netstate, const Float_p input);

Float_t network_get_cost(const NeuronalNetwork_p network, NetState_p netstate, const Float_p labels);
Float_t network_get_accuracy(const NeuronalNetwork_p network, NetState_p netstate, const Float_p labels);
void network_derive_cost(const NeuronalNetwork_p network, NetState_p netstate, const Float_p labels);
void network_gradient_descent(const NeuronalNetwork_p network, NetState_p netstate, Float_t learn_rate);

#endif /* NETWORK_H_INCLUDED */
