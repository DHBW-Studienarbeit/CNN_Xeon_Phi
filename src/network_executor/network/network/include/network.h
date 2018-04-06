#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "settings.h"
#include "datasupplier.h"
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
#define NETWORK_ACTIVATION_SIZE 67390
#define NETWORK_POOLING_MEM_SIZE 9920
#define NETWORK_WEIGHTS_F_SIZE 2654
#define NETWORK_WEIGHTS_I_SIZE 39680
// [[[end]]]


extern Float_t net_activations[NETWORK_ACTIVATION_SIZE];
extern Float_t net_activations_errors[NETWORK_ACTIVATION_SIZE];
extern Int_t net_pooling_mem[NETWORK_POOLING_MEM_SIZE];
extern Float_t net_weights_f[NETWORK_WEIGHTS_F_SIZE];
extern Float_t net_weights_f_errors[NETWORK_WEIGHTS_F_SIZE];
extern Int_t net_weights_i[NETWORK_WEIGHTS_I_SIZE];


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
} NeuronalNetwork_t, *NeuronalNetwork_p

void network_forward(NeuronalNetwork_p network, DataSupplier_p input_supply);
void network_backward(NeuronalNetwork_p network, DataSupplier_p input_supply);

void network_reset_errors(NeuronalNetwork_p network);
void network_gradient_descent(NeuronalNetwork_p network);

#endif /* NETWORK_H_INCLUDED */
