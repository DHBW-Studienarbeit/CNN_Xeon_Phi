#ifndef FULLYCONNECTEDLAYER_H_INCLUDED
#define FULLYCONNECTEDLAYER_H_INCLUDED

#include "layer_commons.h"



typedef struct
{
    // activations
    // relative description of the input activations
    Float_p input_activation_start;
    Float_p input_activation_error_start;
    Int_t input_activation_count;
    // relative description of the output activations
    Float_p output_activation_start;
    Float_p output_activation_error_start;
    Int_t output_activation_count;
    // weights
    // weight dimension
    Int_t single_input_count;
    Int_t single_output_count;
    Int_t batch_count;
    // size of the weight (derivation) section
    Int_t weights_count_total;
    // relative position for weight derivations
    Float_p weights_start;
    Float_p weights_error_start;
    Float_p biases_start;
    Float_p biases_error_start;
    // shared arrays
    Float_p shared_tmp_floats;
    Float_p shared_ones_floats;
} FullyConnectedLayer_t, *FullyConnectedLayer_p;


INLINE void layer_fullcon_forward(  const FullyConnectedLayer_p layerinfo);
INLINE void layer_fullcon_backward( const FullyConnectedLayer_p layerinfo);
INLINE void layer_fullcon_first_forward(const FullyConnectedLayer_p layerinfo,
                                        const Float_p input_start);
INLINE void layer_fullcon_first_backward(const FullyConnectedLayer_p layerinfo,
                                         const Float_p input_start);
INLINE Float_p layer_fullcon_get_output(const FullyConnectedLayer_p layerinfo);
INLINE Int_t layer_fullcon_get_output_position( const FullyConnectedLayer_p layerinfo,
                                                Int_t p,
                                                Int_t y,
                                                Int_t x,
                                                Int_t f);

#endif /*FULLYCONNECTEDLAYER_H_INCLUDED*/
