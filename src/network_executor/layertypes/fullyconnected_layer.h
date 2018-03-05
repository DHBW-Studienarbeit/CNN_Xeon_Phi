#ifndef FULLYCONNECTEDLAYER_H_INCLUDED
#define FULLYCONNECTEDLAYER_H_INCLUDED

#include "settings.h"
#include "tensor_description.h"


typedef struct {
    Layer_t shared_info;
    AbsTensorDescription_t weights;
    AbsTensorDescription_t bias;
    RelTensorDescription_t weights_derivations;
    RelTensorDescription_t bias_derivations;
} FullyConnectedLayer_t, *FullyConnectedLayer_p;



#endif /*FULLYCONNECTEDLAYER_H_INCLUDED*/
