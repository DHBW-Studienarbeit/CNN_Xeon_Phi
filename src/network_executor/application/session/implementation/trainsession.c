#include "trainsession.h"


void exec_trainsession(const NeuronalNetwork_p network, DataSupplier_p supplier, Int_t num_of_batches, Float_t learnrate)
{
    Int_t i;
    for(i=0; i<num_of_batches; i++)
    {
        datasupply_next_batch(supplier);
        network_forward(network, datasupply_get_input(supplier));
        network_derive_cost(network, datasupply_get_output(supplier));
        network_backward(network, datasupply_get_input(supplier));
        network_gradient_descent(network, learnrate);
    }
}
