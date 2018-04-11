#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "settings.h"
#include "fullyconnected_layer.h"
#include "convlayer.h"
#include "maxpoollayer.h"


/* [[[cog
import cog
from network_descriptor.NetInstance import net

cog.outl("#define NETWORK_ACTIVATION_SIZE " + str(net._activation_size))
cog.outl("#define NETWORK_POOLING_MEM_SIZE " + str(net._act_mem_i_size))
cog.outl("#define NETWORK_WEIGHTS_F_SIZE " + str(net._weights_f_size))
cog.outl("#define NETWORK_WEIGHTS_I_SIZE " + str(net._weights_i_size))
]]] */
#define NETWORK_ACTIVATION_SIZE 674004
#define NETWORK_POOLING_MEM_SIZE 99200
#define NETWORK_WEIGHTS_F_SIZE 134464
#define NETWORK_WEIGHTS_I_SIZE 396800
// [[[end]]]


extern Float_t net_activations[NETWORK_ACTIVATION_SIZE];
extern Float_t net_activations_errors[NETWORK_ACTIVATION_SIZE];
extern Int_t net_pooling_mem[NETWORK_POOLING_MEM_SIZE];
extern Float_t net_weights_f[NETWORK_WEIGHTS_F_SIZE];
extern Float_t net_weights_f_errors[NETWORK_WEIGHTS_F_SIZE];
extern const Int_t net_weights_i[NETWORK_WEIGHTS_I_SIZE];


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
    // [[[end]]]
} NeuronalNetwork_t, *NeuronalNetwork_p;

void network_forward(NeuronalNetwork_p network, Float_p input);
void network_backward(NeuronalNetwork_p network, Float_p input);

Float_t network_get_cost(NeuronalNetwork_p network, Float_p labels);
void network_derive_cost(NeuronalNetwork_p network, Float_p labels);
void network_gradient_descent(NeuronalNetwork_p network, Float_t learn_rate);

#endif /* NETWORK_H_INCLUDED */
