#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "settings.h"
#include "fullyconnected_layer.h"
#include "convlayer.h"
#include "maxpoollayer.h"

typedef struct
{

} NeuronalNetwork_t, *NeuronalNetwork_p

void network_forward(NeuronalNetwork_p network, DataSupplier_p input_supply);
void network_backward(NeuronalNetwork_p network, DataSupplier_p input_supply);

#endif /* NETWORK_H_INCLUDED */
