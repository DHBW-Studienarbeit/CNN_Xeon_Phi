#ifndef NETSTATE_H_INCLUDED
#define NETSTATE_H_INCLUDED

#include "settings.h"


/* [[[cog
import cog
from network_descriptor.NetInstance import net

cog.outl("#define NETWORK_ACTIVATION_SIZE " + str(net._activation_size))
cog.outl("#define NETWORK_POOLING_MEM_SIZE " + str(net._act_mem_i_size))
cog.outl("#define NETWORK_WEIGHTS_F_SIZE " + str(net._weights_f_size))
cog.outl("#define NETWORK_WEIGHTS_I_SIZE " + str(net._weights_i_size))
]]] */
#define NETWORK_ACTIVATION_SIZE 320848
#define NETWORK_POOLING_MEM_SIZE 45056
#define NETWORK_WEIGHTS_F_SIZE 1111952
#define NETWORK_WEIGHTS_I_SIZE 180224
// [[[end]]]

typedef struct {
    Float_p activations;
    Float_p activations_errors;
    Float_p weights_f;
    Float_p weights_f_errors;
    Int_p weights_i;
    Int_p pooling_mem;
} NetState_t, *NetState_p;


#endif /* NETSTATE_H_INCLUDED */
