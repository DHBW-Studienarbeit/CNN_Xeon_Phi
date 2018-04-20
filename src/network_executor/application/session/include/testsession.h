#ifndef TESTSESSION_H_INCLUDED
#define TESTSESSION_H_INCLUDED

#include "settings.h"
#include "netstate.h"
#include "network.h"
#include "datasupplier.h"


typedef struct
{
    Float_t cost;
    Float_t accuracy;
} TestResult_t, *TestResult_p;

void exec_testsession(const NeuronalNetwork_p network, DataSupplier_p supplier, Int_t num_of_batches, TestResult_p result);


#endif /* TESTSESSION_H_INCLUDED */
