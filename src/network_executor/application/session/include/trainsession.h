#ifndef TRAINSESSION_H_INCLUDED
#define TRAINSESSION_H_INCLUDED

#include "settings.h"
#include "network.h"
#include "datasupplier.h"

void exec_trainsession(const NeuronalNetwork_p network, DataSupplier_p supplier, Float_t num_of_batches);

#endif /* TRAINSESSION_H_INCLUDED */
