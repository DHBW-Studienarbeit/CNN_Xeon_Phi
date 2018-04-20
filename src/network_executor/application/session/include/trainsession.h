#ifndef TRAINSESSION_H_INCLUDED
#define TRAINSESSION_H_INCLUDED

#include "settings.h"
#include "network.h"
#include "datasupplier.h"

void exec_trainsession(const NeuronalNetwork_p network, DataSupplier_p supplier, Int_t num_of_batches, Float_t learnrate);

#endif /* TRAINSESSION_H_INCLUDED */
