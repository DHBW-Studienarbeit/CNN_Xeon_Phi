#include "testsession.h"
#include "mathematics.h"


void exec_testsession(const NeuronalNetwork_p network, DataSupplier_p supplier, Int_t num_of_batches, TestResult_p result)
{
    result->accuracy = 0.0f;
    result->cost = 0.0f;
    Int_t i;
    for(i=0; i<num_of_batches; i++)
    {
        datasupply_next_batch(supplier);
        network_forward(network, datasupply_get_input(supplier));
        result->accuracy += network_get_accuracy(network, datasupply_get_output(supplier));
        result->cost += network_get_cost(network, datasupply_get_output(supplier));
    }
    result->cost /= num_of_batches;
    result->accuracy /= num_of_batches;
}
