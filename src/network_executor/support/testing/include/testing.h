#ifndef SUPPORT_TESTING_H_INCLUDED
#define SUPPORT_TESTING_H_INCLUDED

#include "settings.h"
#include "netstate.h"
#include "network.h"
#include "datasupplier.h"

void dump_weights(NetState_p netstate);
void dump_output_labels(NeuronalNetwork_p network, NetState_p netstate, DataSupplier_p supplier);
void dump_f_array(const char *name, Int_t count, Float_p data);
void dump_i_array(const char *name, Int_t count, Int_p data);

#endif /* SUPPORT_TESTING_H_INCLUDED */
