#ifndef CONV_LAYER_H_INCLUDED
#define CONV_LAYER_H_INCLUDED

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
    // input shape dimensions
		// used for maxpool shape generation
    Int_t input_x_count;
    Int_t input_xy_count;
    // size of the weight (derivation) section
    Int_t weights_total_count;
    // relative position for weight derivations
    Float_p weights_start;
    Float_p weights_error_start;
    Float_p biases_start;
    Float_p biases_error_start;
    // shared arrays
    Float_p shared_tmp_floats;
    Float_p shared_ones_floats;
    // learn_reduction
    Float_t learn_reduction;
} ConvolutionalLayer_t, *ConvolutionalLayer_p;


INLINE void layer_conv_forward( const ConvolutionalLayer_p layerinfo);
INLINE void layer_conv_backward(const ConvolutionalLayer_p layerinfo);
INLINE void layer_conv_first_forward(const ConvolutionalLayer_p layerinfo,
                                     const Float_p input_start);
INLINE void layer_conv_first_backward(const ConvolutionalLayer_p layerinfo,
                                      const Float_p input_start);
INLINE Float_p layer_conv_get_output(const ConvolutionalLayer_p layerinfo);
INLINE Int_t layer_conv_get_output_position(const ConvolutionalLayer_p layerinfo,
                                         Int_t p,
                                         Int_t y,
                                         Int_t x,
                                         Int_t f);


#endif /*CONV_LAYER_H_INCLUDED*/
