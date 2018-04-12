#include "settings.h"
#include "layer_commons.h"
#include "network.h"



void netstate_init(NetState_p netstate)
{
    netstate->activations = net_activations;
    netstate->activations_errors = net_activations_errors;
    netstate->weights_f = net_weights_f;
    netstate->weights_f_errors = net_weights_f_errors;
    netstate->weights_i = net_weights_i;
    netstate->pooling_mem = net_pooling_mem;
}
