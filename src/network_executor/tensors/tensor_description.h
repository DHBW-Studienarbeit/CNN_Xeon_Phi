#ifndef TENSOR_DESCRIPTION_H_INCLUDED
#define TENSOR_DESCRIPTION_H_INCLUDED

#include "settings.h"
#include "cnn_fracted_vector.h"

#define NUM_TENSORDIMS  3;



typedef struct {
    FractedVector_t base_vector;
    Index_t dim[NUM_TENSORDIMS];
} RelTensor_t, *RelTensor_p;

typedef struct {
    Float_p startpoint;
    RelTensor_p description;
} AbsTensor_t, *AbsTensor_p;



Float_t rel_tensor_get(RelTensor_p tensor, Float_p reference, Index_t index, Index_t index, Index_t index);
Float_p rel_tensor_get(RelTensor_p tensor, Float_p reference, Index_t index, Index_t index, Index_t index);

Float_t abs_tensor_get(AbsTensor_p tensor, Float_p reference, Index_t index, Index_t index, Index_t index);
Float_p abs_tensor_get(AbsTensor_p tensor, Float_p reference, Index_t index, Index_t index, Index_t index);

void get_part_of_fracted_vector(FractedVector_p origin, FractedVector_p dest, FractedVector_p modification);

#endif /*TENSOR_DESCRIPTION_H_INCLUDED*/
