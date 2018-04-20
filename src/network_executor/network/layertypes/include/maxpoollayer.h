#ifndef MAXPOOLING_LAYER_H_INCLUDED
#define MAXPOOLING_LAYER_H_INCLUDED


#include "layer_commons.h"


// Layout defines which inputs have to be compared for each output
typedef struct
{
    Int_t relevant_entries_count;
    Int_t num_of_lines;
    Int_t relevant_columns_per_line;
    Int_p relevant_columns_start;
} PoolingTableLayout_t, *PoolingTableLayout_p;

// subset of PoolingTableLayout
typedef struct
{
    Int_t relevant_entries_count;
    Int_p relevant_entries_start;
} PoolingTableDefinite_t, *PoolingTableDefinite_p;


typedef struct
{
    // activations
    // relative description of the input activations
    //Int_t input_activation_offset;
    Float_p input_activation_start;
    Float_p input_activation_error_start;
    Int_t input_activation_count;
    // relative description of the output activations
    //Int_t output_activation_offset;
    Float_p output_activation_start;
    Float_p output_activation_error_start;
    Int_t output_activation_count;
    // shape
        // used for maxpool weight generation
    Int_t output_p;
    Int_t output_y;
    Int_t output_x;
    Int_t output_f;
    // weights
    PoolingTableLayout_t pooling_layout;
    PoolingTableDefinite_t pooling_mem;
} MaxPoolingLayer_t, *MaxPoolingLayer_p;


INLINE void layer_maxpool_forward(  const MaxPoolingLayer_p layerinfo);
INLINE void layer_maxpool_backward( const MaxPoolingLayer_p layerinfo);
INLINE void layer_maxpool_first_forward(const MaxPoolingLayer_p layerinfo,
                                        const Float_p input_start);
INLINE void layer_maxpool_first_backward(   const MaxPoolingLayer_p layerinfo,
                                            const Float_p input_start);
INLINE Float_p layer_maxpool_get_output(const MaxPoolingLayer_p layerinfo);
INLINE Int_t layer_maxpool_get_output_position( const MaxPoolingLayer_p layerinfo,
                                                Int_t p,
                                                Int_t y,
                                                Int_t x,
                                                Int_t f);


#endif /*MAXPOOLING_LAYER_H_INCLUDED*/
