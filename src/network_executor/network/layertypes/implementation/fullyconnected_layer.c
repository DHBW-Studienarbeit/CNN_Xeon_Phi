#include "fullyconnected_layer.h"
#include "shared_arrays.h"
#include "mathematics.h"


INLINE void layer_fullcon_forward(  const FullyConnectedLayer_p layerinfo,
                                    NetState_p netstate
                                 )
{
    Float_p z_vector_start = shared_tmp_floats;
    // multiply input batch with weight matrix
    MATH_MULT_MAT_MAT(  CblasColMajor,
                        CblasTrans,
                        CblasNoTrans,
                        layerinfo->single_output_count,
                        layerinfo->batch_count,
                        layerinfo->single_input_count,
                        1.0f,
                        netstate->weights_f + layerinfo->weights_offset,
                        layerinfo->single_input_count,
                        netstate->activations + layerinfo->input_activation_offset,
                        layerinfo->single_input_count,
                        0.0f,
                        z_vector_start,
                        layerinfo->single_output_count
                    ) ;
    // copy and concatenate bias vector; add it to weighted inputs
    MATH_MULT_MAT_MAT(  CblasColMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        layerinfo->single_output_count,
                        layerinfo->batch_count,
                        1,
                        1.0f,
                        netstate->weights_f + layerinfo->biases_offset,
                        layerinfo->single_output_count,
                        shared_ones_floats,
                        1,
                        1.0f,
                        z_vector_start,
                        layerinfo->single_output_count
                    ) ;
    // apply sigmoid to result
    sigmoid(    layerinfo->output_activation_count,
                z_vector_start,
                netstate->activations + layerinfo->output_activation_offset
            ) ;
}

INLINE void layer_fullcon_backward( const FullyConnectedLayer_p layerinfo,
                                    NetState_p netstate
                                  )
{
    Float_p y_deriv_z = shared_tmp_floats;
    Float_p rev_sigmoid_buffer = shared_tmp_floats + layerinfo->output_activation_count;
    Float_p cost_deriv_z = rev_sigmoid_buffer;
    // calc y_deriv_z from y
    sigmoid_derivation( layerinfo->output_activation_count,
                        netstate->activations + layerinfo->output_activation_offset,
                        y_deriv_z,
                        rev_sigmoid_buffer
                      );
    // calc cost_deriv_z from y_deriv_z and cost_deriv_y
    MATH_VECT_MUL(  layerinfo->output_activation_count,
                    y_deriv_z,
                    netstate->activations_errors + layerinfo->output_activation_offset,
                    cost_deriv_z
                 );
    // calc cost_deriv_x from cost_deriv_z and weights
    MATH_MULT_MAT_MAT(  CblasColMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        layerinfo->single_input_count,
                        layerinfo->batch_count,
                        layerinfo->single_output_count,
                        1.0f,
                        netstate->weights_f + layerinfo->weights_offset,
                        layerinfo->single_input_count,
                        cost_deriv_z,
                        layerinfo->single_output_count,
                        0.0f,
                        netstate->activations_errors + layerinfo->input_activation_offset,
                        layerinfo->single_input_count
                     );
    // add cost_deriv_z to cost_deriv_bias for all datasets of the batch
    MATH_MULT_MAT_VECT( CblasColMajor,
                        CblasNoTrans,
                        layerinfo->single_output_count,
                        layerinfo->batch_count,
                        1.0f,
                        cost_deriv_z,
                        layerinfo->single_output_count,
                        shared_ones_floats,
                        1,
                        0.0f,
                        netstate->weights_f_errors + layerinfo->biases_offset,
                        1
                    );
    // calc cost_deriv_weights
    MATH_MULT_MAT_MAT(  CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        layerinfo->single_output_count,
                        layerinfo->single_input_count,
                        layerinfo->batch_count,
                        1.0f,
                        cost_deriv_z,
                        layerinfo->single_output_count,
                        netstate->activations + layerinfo->input_activation_offset,
                        layerinfo->single_input_count,
                        0.0f,
                        netstate->weights_f_errors + layerinfo->weights_offset,
                        layerinfo->single_input_count
                     );
}

INLINE void layer_fullcon_first_forward(const FullyConnectedLayer_p layerinfo,
                                        NetState_p netstate,
                                        const Float_p input_start
                                       )
{
    Float_p z_vector_start = shared_tmp_floats;
    // multiply input batch with weight matrix
    MATH_MULT_MAT_MAT(  CblasColMajor,
                        CblasTrans,
                        CblasNoTrans,
                        layerinfo->single_output_count,
                        layerinfo->batch_count,
                        layerinfo->single_input_count,
                        1.0f,
                        netstate->weights_f + layerinfo->weights_offset,
                        layerinfo->single_input_count,
                        input_start,
                        layerinfo->single_input_count,
                        0.0f,
                        z_vector_start,
                        layerinfo->single_output_count
                    ) ;
    // copy and concatenate bias vector; add it to weighted inputs
    MATH_MULT_MAT_MAT(  CblasColMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        layerinfo->single_output_count,
                        layerinfo->batch_count,
                        1,
                        1.0f,
                        netstate->weights_f + layerinfo->biases_offset,
                        layerinfo->single_output_count,
                        shared_ones_floats,
                        1,
                        1.0f,
                        z_vector_start,
                        layerinfo->single_output_count
                    ) ;
    // apply sigmoid to result
    sigmoid(    layerinfo->output_activation_count,
                z_vector_start,
                netstate->activations + layerinfo->output_activation_offset
            ) ;
}

INLINE void layer_fullcon_first_backward(const FullyConnectedLayer_p layerinfo,
                                         NetState_p netstate,
                                         const Float_p input_start
                                        )
{
    Float_p y_deriv_z = shared_tmp_floats;
    Float_p rev_sigmoid_buffer = shared_tmp_floats + layerinfo->output_activation_count;
    Float_p cost_deriv_z = rev_sigmoid_buffer;
    // calc y_deriv_z from y
    sigmoid_derivation( layerinfo->output_activation_count,
                        netstate->activations + layerinfo->output_activation_offset,
                        y_deriv_z,
                        rev_sigmoid_buffer
                      );
    // calc cost_deriv_z from y_deriv_z and cost_deriv_y
    MATH_VECT_MUL(  layerinfo->output_activation_count,
                    y_deriv_z,
                    netstate->activations_errors + layerinfo->output_activation_offset,
                    cost_deriv_z
                 );
    // add cost_deriv_z to cost_deriv_bias for all datasets of the batch
    MATH_MULT_MAT_VECT( CblasColMajor,
                        CblasNoTrans,
                        layerinfo->single_output_count,
                        layerinfo->batch_count,
                        1.0f,
                        cost_deriv_z,
                        layerinfo->single_output_count,
                        shared_ones_floats,
                        1,
                        0.0f,
                        netstate->weights_f_errors + layerinfo->biases_offset,
                        1
                    );
    // calc cost_deriv_weights
    MATH_MULT_MAT_MAT(  CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        layerinfo->single_output_count,
                        layerinfo->single_input_count,
                        layerinfo->batch_count,
                        1.0f,
                        cost_deriv_z,
                        layerinfo->single_output_count,
                        input_start,
                        layerinfo->single_input_count,
                        0.0f,
                        netstate->weights_f_errors + layerinfo->weights_offset,
                        layerinfo->single_input_count
                     );
}

INLINE Float_p layer_fullcon_get_output(const FullyConnectedLayer_p layerinfo,
                                        NetState_p netstate
                                       )
{
    return netstate->activations + layerinfo->output_activation_offset;
}


INLINE Int_t layer_fullcon_get_output_position( const FullyConnectedLayer_p layerinfo,
                                                NetState_p netstate,
                                                Int_t p,
                                                Int_t y,
                                                Int_t x,
                                                Int_t f
                                              )
{
    return f + p * layerinfo->single_output_count;
}
