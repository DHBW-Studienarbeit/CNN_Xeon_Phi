#include "testsession.h"
#include "mathematics.h"


Float_t exec_testsession(const NeuronalNetwork_p network, NetState_p netstate, DataSupplier_p supplier, Int_t num_of_batches)
{
    Float_t sum_accuracy=0.0f;
    Int_t i;
    for(i=0; i<num_of_batches; i++)
    {
        datasupply_next_batch(supplier);
        network_forward(network, netstate, datasupply_get_input(supplier));
        sum_accuracy += network_get_accuracy(network, netstate, datasupply_get_output(supplier));
    }
    return sum_accuracy / num_of_batches;
}
