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
cog.outl("#define NETWORK_POOLING_LAYOUT_SIZE " + str(net._weights_i_size))
cog.outl("#define SHARED_ARRAY_SIZE " + str(net._max_act_count))
cog.outl("#define SHARED_TMP_ARRAY_SIZE " + str(2 * net._max_act_count))
]]] */
#define NETWORK_ACTIVATION_SIZE 648736
#define NETWORK_POOLING_MEM_SIZE 90112
#define NETWORK_WEIGHTS_F_SIZE 1111968
#define NETWORK_POOLING_LAYOUT_SIZE 360448
#define SHARED_ARRAY_SIZE 397760
#define SHARED_TMP_ARRAY_SIZE 795520
// [[[end]]]


typedef struct
{
    Float_p activations;
    Float_p activations_errors;
    Float_p weights_f;
    Float_p weights_f_errors;
    Int_p pooling_layout;
    Int_p pooling_mem;

    Float_p shared_ones_floats;
    Float_p shared_tmp_floats;
    Int_p shared_tmp_ints;

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


void network_init(const NeuronalNetwork_p network);

void network_forward(const NeuronalNetwork_p network, const Float_p input);
void network_backward(const NeuronalNetwork_p network, const Float_p input);

Float_t network_get_cost(const NeuronalNetwork_p network, const Float_p labels);
Float_t network_get_accuracy(const NeuronalNetwork_p network, const Float_p labels);
void network_derive_cost(const NeuronalNetwork_p network, const Float_p labels);
void network_gradient_descent(const NeuronalNetwork_p network, Float_t learn_rate);

#endif /* NETWORK_H_INCLUDED */
