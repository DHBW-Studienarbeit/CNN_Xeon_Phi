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
    Int_t relevant_columns[MAXPOOLING_LAYOUT_MAX_ELEMENT_COUNT];
} PoolingTableLayout_t, *PoolingTableLayout_p;

// subset of PoolingTableLayout
typedef struct
{
    Int_t relevant_entries_count;
    Int_t relevant_columns[MAXPOOLING_LAYOUT_MAX_ELEMENT_COUNT];
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
    // weight dimension
    Int_t feature_count;
    Int_t input_x_count;
    Int_t input_y_count;
    Int_t filter_x_count;
    Int_t filter_y_count;
    Int_t batch_count;
    // size of the weight section
    // note that in this layertype weightdata is stored as integer
    Int_t weight_layout_total_count;
    Int_t weight_layout_colums_per_line
    // relative position for weight derivations
    Int_t weight_realisation_offset;
    Int_t weights_realisation_total_count;
    // absolute position of the weights
    Int_p weight_layout_start;
} MaxPoolingLayer_t, *MaxPoolingLayer_p


INLINE void layer_maxpool_forward(const MaxPoolingLayer_p layerinfo, Float_p activations_start);
INLINE void layer_maxpool_backward(const MaxPoolingLayer_p layerinfo, Float_p activations_start, Float_p activations_deriv_start, Float_p weight_errors_start);

INLINE void layer_maxpool_first_forward(const MaxPoolingLayer_p layerinfo, Float_p activations_start, Float_p input_start);
INLINE void layer_maxpool_first_backward(const MaxPoolingLayer_p layerinfo, Float_p activations_start, Float_p input_start, Float_p activations_deriv_start, Float_p weight_errors_start);

INLINE Float_p layer_maxpool_get_output(const MaxPoolingLayer_p layerinfo, Float_p activations_start);

#endif /*MAXPOOLING_LAYER_H_INCLUDED*/
