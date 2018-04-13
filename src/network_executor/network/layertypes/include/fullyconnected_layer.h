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
} FullyConnectedLayer_t, *FullyConnectedLayer_p;


INLINE void layer_fullcon_forward(  const FullyConnectedLayer_p layerinfo,
                                    NetState_p netstate);
INLINE void layer_fullcon_backward( const FullyConnectedLayer_p layerinfo,
                                    NetState_p netstate);
INLINE void layer_fullcon_first_forward(const FullyConnectedLayer_p layerinfo,
                                        NetState_p netstate,
                                        const Float_p input_start);
INLINE void layer_fullcon_first_backward(const FullyConnectedLayer_p layerinfo,
                                         NetState_p netstate,
                                         const Float_p input_start);
INLINE Float_p layer_fullcon_get_output(const FullyConnectedLayer_p layerinfo,
                                        NetState_p netstate);
INLINE Int_t layer_fullcon_get_output_position( const FullyConnectedLayer_p layerinfo,
                                                NetState_p netstate,
                                                Int_t p,
                                                Int_t y,
                                                Int_t x,
                                                Int_t f);

#endif /*FULLYCONNECTEDLAYER_H_INCLUDED*/
