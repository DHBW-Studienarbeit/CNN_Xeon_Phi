#ifndef TESTSESSION_H_INCLUDED
#define TESTSESSION_H_INCLUDED

#include "settings.h"
#include "network.h"
#include "datasupplier.h"

Float_t exec_testsession(const NeuronalNetwork_p network, DataSupplier_p supplier, Int_t num_of_batches);


#endif /* TESTSESSION_H_INCLUDED */
