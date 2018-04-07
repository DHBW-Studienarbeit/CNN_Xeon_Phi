#ifndef CONV_LAYER_H_INCLUDED
#define CONV_LAYER_H_INCLUDED

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
    Int_t filter_feature_input_count;
    Int_t filter_x_count;
    Int_t filter_y_count;
    Int_t filter_feature_output_count;
    Int_t batch_count;
    // further matrix dimensions
    Int_t filter_matrix_width;
    Int_t input_matrix_height;
    Int_t input_matrix_width;
    Int_t input_matrix_toplayer_elements_count;
    Int_t partial_output_matrix_count;
    Int_t full_output_matrix_width;
    // partial matrix dimensions
		// seems to be unused
    //Int_t filter_x_pos_count;
    //Int_t filter_y_pos_count;
    // size of the weight (derivation) section
    Int_t weights_total_count;
    // relative position for weight derivations
    Int_t weights_offset;
    Int_t biases_offset;
    // absolute position of the weights
    Float_p weights_start;
    Float_p biases_start;
} ConvolutionalLayer_t, *ConvolutionalLayer_p;


INLINE void layer_conv_forward(const ConvolutionalLayer_p layerinfo, Float_p activations_start);
INLINE void layer_conv_backward(const ConvolutionalLayer_p layerinfo, Float_p activations_start, Float_p activations_deriv_start, Float_p weight_errors_start);

INLINE void layer_conv_first_forward(const ConvolutionalLayer_p layerinfo, Float_p activations_start, Float_p input_start);
INLINE void layer_conv_first_backward(const ConvolutionalLayer_p layerinfo, Float_p activations_start, Float_p input_start, Float_p activations_deriv_start, Float_p weight_errors_start);

INLINE Float_p layer_conv_get_output(const ConvolutionalLayer_p layerinfo, Float_p activations_start);


#endif /*CONV_LAYER_H_INCLUDED*/
