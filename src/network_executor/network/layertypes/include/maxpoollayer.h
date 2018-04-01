#ifndef MAXPOOLING_LAYER_H_INCLUDED
#define MAXPOOLING_LAYER_H_INCLUDED


#include "layer_commons.h"
#include "shared_arrays.h"

#define MAXPOOLING_LAYOUT_MAX_ELEMENT_COUNT SHARED_ARRAY_SIZE

// pooling matrix is stored in the CSR format (zero-based)
typedef struct
{
    Int_t relevant_entries_count;
    Int_t num_of_lines;
    Int_t relevant_columns_per_line;
    Int_p relevant_columns;
} PoolingTableLayout_t, *PoolingTableLayout_p;

// subset of PoolingTableLayout
typedef struct
{
    Int_t relevant_entries_count;
    Int_t relevant_columns_offset;
} PoolingTableDefinite_t, *PoolingTableDefinite_p;


typedef struct
{
    // activations
    // relative description of the input activations
    Int_t input_activation_offset;
    Int_t input_activation_count;
    // relative description of the output activations
    Int_t output_activation_offset;
    Int_t output_activation_count;
    // weights
    PoolingTableLayout_t pooling_layout;
    PoolingTableDefinite_t weight_shape;
} MaxPoolingLayer_t, *MaxPoolingLayer_p


INLINE void layer_maxpool_forward(const MaxPoolingLayer_p layerinfo, Float_p activations_start, const Int_p int_weights_start);
INLINE void layer_maxpool_backward(const MaxPoolingLayer_p layerinfo, Float_p activations_start, const Int_p int_weights_start, Float_p activations_deriv_start, Float_p weight_errors_start);

INLINE void layer_maxpool_first_forward(const MaxPoolingLayer_p layerinfo, Float_p activations_start, const Int_p int_weights_start, Float_p input_start);
INLINE void layer_maxpool_first_backward(const MaxPoolingLayer_p layerinfo, Float_p activations_start, const Int_p int_weights_start, Float_p input_start, Float_p activations_deriv_start, Float_p weight_errors_start);

INLINE Float_p layer_maxpool_get_output(const MaxPoolingLayer_p layerinfo, Float_p activations_start);

#endif /*MAXPOOLING_LAYER_H_INCLUDED*/
