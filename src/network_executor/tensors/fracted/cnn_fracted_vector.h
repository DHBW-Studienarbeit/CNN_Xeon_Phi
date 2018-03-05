#ifndef CNN_TENSOR_FRACTED_VECTOR_H_INCLUDED
#define CNN_TENSOR_FRACTED_VECTOR_H_INCLUDED


#include "settings.h"


typedef struct {
    Index_t offset;
    Index_t size;
    Index_t stepsize;
    Index_t gapsize;
} FractedVector_t, *FractedVektorDescription_p;


Float_t fracted_vector_get(FractedVector_p vector, Float_p reference, Index_t index);
Float_p fracted_vector_get_address(FractedVector_p vector, Float_p reference, Index_t index);
Index_t fracted_vector_get_offset(FractedVector_p vector, Index_t index);


#endif /*CNN_TENSOR_FRACTED_VECTOR_H_INCLUDED*/
