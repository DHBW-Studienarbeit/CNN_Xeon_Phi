#include "maxpoollayer.h"
#include "mathematics.h"


static inline Int_t get_poolshape_entry(const MaxPoolingLayer_p layerinfo,
                                        Int_t output_index,
                                        Int_t column_index
                                       )
{
    return layerinfo->pooling_layout.relevant_columns_start[
        output_index * layerinfo->pooling_layout.relevant_columns_per_line
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
        Float_t current_value;
        Float_t max_value;
        Int_t current_index;
        Int_t max_index = get_poolshape_entry(layerinfo, i, 0);
        max_value = layerinfo->input_activation_start[max_index];
        // find max relevant input
        for(current_index=1; current_index < layerinfo->pooling_layout.relevant_columns_per_line; current_index++)
        {
            current_index = get_poolshape_entry(layerinfo, i, current_index);
            current_value = layerinfo->input_activation_start[current_index];
            if(current_value > max_value)
            {
                max_value = current_value;
                max_index = current_index;
            }
        }
        // write max relevant input to output activation and pooling_mem(for reusing it during backpropagation)
        layerinfo->pooling_mem.relevant_entries_start[i] = max_index;
        layerinfo->output_activation_start[i] = max_value;
    }
}

INLINE void layer_maxpool_backward(const MaxPoolingLayer_p layerinfo)
{
    Int_t i;
    // prepare by setting all input errors to zero
    #pragma omp parallel for
    for(i=0; i<layerinfo->input_activation_count; i++)
    {
        layerinfo->input_activation_error_start[i] = 0.0f;
    }
    // for each output of the whole batch (parallel)
    #pragma omp parallel for
    for(i=0; i<layerinfo->output_activation_count; i++)
    {
        // copy the output error to the responsible input error (position found in pooling_mem)
        layerinfo->input_activation_error_start[layerinfo->pooling_mem.relevant_entries_start[i]]
         = layerinfo->output_activation_error_start[i];
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
        Float_t current_value;
        Float_t max_value;
        Int_t current_index;
        Int_t max_index = get_poolshape_entry(layerinfo, i, 0);
        max_value = layerinfo->input_activation_start[max_index];
        // find max relevant input
        for(current_index=1; current_index < layerinfo->pooling_layout.relevant_columns_per_line; current_index++)
        {
            current_index = get_poolshape_entry(layerinfo, i, current_index);
            current_value = layerinfo->input_activation_start[current_index];
            if(current_value > max_value)
            {
                max_value = current_value;
                max_index = current_index;
            }
        }
        // write max relevant input to output activation; pooling_mem not needed
        layerinfo->output_activation_start[i] = max_value;
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
    return layerinfo->output_activation_start;
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
