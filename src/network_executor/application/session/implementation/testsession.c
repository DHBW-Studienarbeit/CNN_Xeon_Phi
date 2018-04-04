#include "testsession.h"

void exec_testsession(NeuronalNetwork_p network, DataSupplier_p supplier)
{
    
    network_forward(network, supplier);
    network_backward(network, supplier);
}
