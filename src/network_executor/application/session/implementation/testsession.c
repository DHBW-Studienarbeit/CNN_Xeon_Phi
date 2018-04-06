#include "testsession.h"

Float_t exec_testsession(NeuronalNetwork_p network, DataSupplier_p supplier, Float_t num_of_batches)
{
    Float_t mean_cost=0.0f;
    Float_t current_cost=0.0f;
    for(Int_t i=0; i<num_of_batches; i++)
    {
        datasupply_next_batch(supplier);
        network_forward(network, datasupply_get_input(supplier));
        current_cost = network_get_cost(network, datasupply_get_output(supplier));
        mean_cost = mean_cost * ((Float_t)i) + current_cost;
        mean_cost /= (Float_t)(i+1);
    }
    return mean_cost;
}
