#ifndef LAYER_COMMON_H_INCLUDED
#define LAYER_COMMON_H_INCLUDED

#include "settings.h"


typedef struct {
    Float_p activations;
    Float_p activations_errors;
    Float_p weights_f;
    Float_p weights_f_errors;
    Int_p weights_i;
    Int_p pooling_mem;
} NetState_t, *NetState_p;


#endif /*LAYER_COMMON_H_INCLUDED*/
