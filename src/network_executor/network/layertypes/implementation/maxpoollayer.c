#include "maxpoollayer.h"
#include "shared_arrays.h"
#include "mathematics.h"


static inline Int_t get_poolshape_entry(const MaxPoolingLayer_p layerinfo,
                                        const Int_p int_weights_start,
                                        Int_t output_index,
                                        Int_t column_index
                                       )
{
    return layerinfo->input_activation_offset
     + int_weights_start[layerinfo->pooling_layout.relevant_columns_offset
         + output_index*(layerinfo->pooling_layout.relevant_columns_per_line)
         + column_index ];
}


INLINE void layer_maxpool_forward(const MaxPoolingLayer_p layerinfo)
{
    Int_t i;
    // for each output of the whole batch (parallel)
    #pragma omp parallel for
    for(i=0; i<layerinfo->output_activation_count; i++)
    {
        // initialize with first relevant input
        Int_t current_index;
        Int_t max_index = 0;
        Float_t current_value;
        Float_t max_value = netstate->activations[get_poolshape_entry(layerinfo, netstate->weights_i, i, 0)];
        // find max relevant input
        for(current_index=1; current_index < layerinfo->pooling_layout.relevant_columns_per_line; current_index++)
        {
            current_value = netstate->activations[get_poolshape_entry(layerinfo, netstate->weights_i, i, current_index)];
            if(current_value > max_value)
            {
                max_value = current_value;
                max_index = current_index;
            }
        }
        // write max relevant input to output activation and pooling_mem(for reusing it during backpropagation)
        netstate->pooling_mem[layerinfo->weight_shape.relevant_entries_offset + i] = get_poolshape_entry(layerinfo, netstate->weights_i, i, max_index);
        netstate->activations[layerinfo->output_activation_offset + i] = max_value;
    }
}

INLINE void layer_maxpool_backward(const MaxPoolingLayer_p layerinfo)
{
    Int_t i;
    // prepare by setting all input errors to zero
    #pragma omp parallel for
    for(i=0; i<layerinfo->input_activation_count; i++)
    {
        netstate->activations_errors[layerinfo->input_activation_offset + i] = 0.0f;
    }
    // for each output of the whole batch (parallel)
    #pragma omp parallel for
    for(i=0; i<layerinfo->output_activation_count; i++)
    {
        // copy the output error to the responsible input error (position found in pooling_mem)
        netstate->activations_errors[ netstate->pooling_mem[layerinfo->weight_shape.relevant_entries_offset + i] ]
         = netstate->activations_errors[layerinfo->output_activation_offset + i];
    }
}

INLINE void layer_maxpool_first_forward(const MaxPoolingLayer_p layerinfo,
                                        const Float_p input_start
                                       )
{
    Int_t i;
    // for each output of the whole batch (parallel)
    #pragma omp parallel for
    for(i=0; i<layerinfo->output_activation_count; i++)
    {
        // initialize with first relevant input
        Int_t max_index = 0;
        Int_t current_index;
        Float_t current_value;
        Float_t max_value = input_start[get_poolshape_entry(layerinfo, netstate->weights_i, i, 0)];
        // find max relevant input
        for(current_index=1; current_index < layerinfo->pooling_layout.relevant_columns_per_line; current_index++)
        {
            current_value = input_start[get_poolshape_entry(layerinfo, netstate->weights_i, i, current_index)];
            if(current_value > max_value)
            {
                max_value = current_value;
                max_index = current_index;
            }
        }
        // write max relevant input to output activation; pooling_mem not needed
        netstate->activations[layerinfo->output_activation_offset + i] = max_value;
    }
}

INLINE void layer_maxpool_first_backward(   const MaxPoolingLayer_p layerinfo,
                                            const Float_p input_start
                                        )
{
    // nothing to do here
}

INLINE Float_p layer_maxpool_get_output(const MaxPoolingLayer_p layerinfo)
{
    return netstate->activations + layerinfo->output_activation_offset;
}

INLINE Int_t layer_maxpool_get_output_position( const MaxPoolingLayer_p layerinfo,
                                                Int_t p,
                                                Int_t y,
                                                Int_t x,
                                                Int_t f
                                              )
{
    return f + x * layerinfo->output_f
     + y * layerinfo->output_f * layerinfo->output_x
     + p * layerinfo->output_f * layerinfo->output_x * layerinfo->output_y;
}
