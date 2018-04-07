#ifndef FULLYCONNECTEDLAYER_H_INCLUDED
#define FULLYCONNECTEDLAYER_H_INCLUDED

#include "layer_commons.h"



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
    Int_t single_input_count;
    Int_t single_output_count;
    Int_t batch_count;
    // size of the weight (derivation) section
    Int_t weights_count_total;
    // relative position for weight derivations
    Int_t weights_offset;
    Int_t biases_offset;
    // absolute position of the weights
    Float_p weights_start;
    Float_p biases_start;
} FullyConnectedLayer_t, *FullyConnectedLayer_p;


INLINE void layer_fullcon_forward(const FullyConnectedLayer_p layerinfo, Float_p activations_start);
INLINE void layer_fullcon_backward(const FullyConnectedLayer_p layerinfo, Float_p activations_start, Float_p activations_deriv_start, Float_p weight_errors_start);

INLINE void layer_fullcon_first_forward(const FullyConnectedLayer_p layerinfo, Float_p activations_start, Float_p input_start);
INLINE void layer_fullcon_first_backward(const FullyConnectedLayer_p layerinfo, Float_p activations_start, Float_p input_start, Float_p activations_deriv_start, Float_p weight_errors_start);

INLINE Float_p layer_fullcon_get_output(const FullyConnectedLayer_p layerinfo, Float_p activations_start);

#endif /*FULLYCONNECTEDLAYER_H_INCLUDED*/
