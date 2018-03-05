#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "settings.h"
#include "tensor_description.h"


typedef struct {
    RelTensorDescription_t activations;
    RelTensorDescription_t input_derivations;
} Layer_t, *Layer_p;



#endif /*LAYER_H_INCLUDED*/
