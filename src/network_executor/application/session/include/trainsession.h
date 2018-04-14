#ifndef TRAINSESSION_H_INCLUDED
#define TRAINSESSION_H_INCLUDED

#include "settings.h"
#include "layer_commons.h"
#include "network.h"
#include "datasupplier.h"

void exec_trainsession(const NeuronalNetwork_p network, NetState_p netstate, DataSupplier_p supplier, Int_t num_of_batches, Float_t learnrate);

#endif /* TRAINSESSION_H_INCLUDED */
