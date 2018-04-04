#include "network.h"

/* [[[cog
import cog
from network_descriptor.NetInstance import net

cog.outl(net._activation_size)


]]] */
// [[[end]]]


void network_forward(NeuronalNetwork_p network, DataSupplier_p input_supply)
{
    /* [[[cog
    import cog
    cog.outl("Hello world")
    ]]] */
    // [[[end]]]
}

void network_backward(NeuronalNetwork_p network, DataSupplier_p input_supply)
{

}

void network_reset_errors(NeuronalNetwork_p network)
{

}

void network_gradient_descent(NeuronalNetwork_p network)
{

}
