#include "convlayer.h"
#include "shared_arrays.h"
#include "mathematics.h"


INLINE void layer_conv_forward( const ConvolutionalLayer_p layerinfo,
                                NetState_p netstate
                              )
{
    Float_p z_vector_start = shared_tmp_floats;
    // matrix products are summed up, but cleaned in the beginning
    Float_t beta=0.0f;
    Int_t i, j;
    for(j=0; j<layerinfo->filter_y_count; j++)
    {
        for(i=0; i<layerinfo->filter_x_count; i++)
        {
            MATH_MULT_MAT_MAT(  CblasColMajor,
                                CblasTrans,
                                CblasNoTrans,
                                layerinfo->filter_feature_output_count,
                                layerinfo->input_matrix_width,
                                layerinfo->input_matrix_height,
                                1.0f,
                                netstate->weights_f + layerinfo->weights_offset + j * layerinfo->input_matrix_height,
                                layerinfo->filter_matrix_width,
                                netstate->activations
                                 + layerinfo->input_activation_offset
                                 + i * layerinfo->filter_feature_input_count
                                 + j * layerinfo->input_matrix_toplayer_elements_count,
                                layerinfo->input_matrix_height,
                                beta,
                                z_vector_start + i * layerinfo->partial_output_matrix_count,
                                layerinfo->filter_feature_output_count
                             );
        }
        beta=1.0f;
    }
    // copy and concatenate bias vector; add it to weighted inputs
    MATH_MULT_MAT_MAT(  CblasColMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        layerinfo->filter_feature_output_count,
                        layerinfo->input_matrix_width,
                        1,
                        1.0f,
                        netstate->weights_f + layerinfo->biases_offset,
                        layerinfo->filter_feature_output_count,
                        shared_ones_floats,
                        1,
                        1.0f,
                        z_vector_start,
                        layerinfo->filter_feature_output_count
                    ) ;
    // apply sigmoid to result
    sigmoid(    layerinfo->output_activation_count,
                z_vector_start,
                netstate->activations + layerinfo->output_activation_offset
            ) ;
}

INLINE void layer_conv_backward(const ConvolutionalLayer_p layerinfo,
                                NetState_p netstate
                               )
{
    Float_p y_deriv_z = shared_tmp_floats;
    Float_p rev_sigmoid_buffer = shared_tmp_floats + layerinfo->output_activation_count;
    Float_p cost_deriv_z = rev_sigmoid_buffer;
    Int_t i, j;
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
    // matrix products are summed up, but cleaned in the beginning
    Float_t beta=0.0f;
    for(j=0; j<layerinfo->filter_y_count; j++)
    {
        for(i=0; i<layerinfo->filter_x_count; i++)
        {
        MATH_MULT_MAT_MAT(  CblasColMajor,
                            CblasNoTrans,
                            CblasNoTrans,
                            layerinfo->input_matrix_height,
                            layerinfo->input_matrix_width,
                            layerinfo->filter_feature_output_count,
                            1.0f,
                            netstate->weights_f + layerinfo->weights_offset + j * layerinfo->input_matrix_height,
                            layerinfo->filter_matrix_width,
                            cost_deriv_z + i * layerinfo->partial_output_matrix_count,
                            layerinfo->filter_feature_output_count,
                            beta,
                            netstate->activations_errors
                            + layerinfo->input_activation_offset
                            + i * layerinfo->filter_feature_input_count
                            + j * layerinfo->input_matrix_toplayer_elements_count,
                            layerinfo->input_matrix_height
                      );
        }
        beta=1.0f;
    }

    // learn reduction factor
    // conv weights are used much more often than std weights
    // learn speed per use must be decreased
    Float_t learn_reduction = 1.0f / (layerinfo->full_output_matrix_width);

    // add cost_deriv_z to cost_deriv_bias for all datasets of the batch
    MATH_MULT_MAT_VECT( CblasColMajor,
                        CblasNoTrans,
                        layerinfo->filter_feature_output_count,
                        layerinfo->full_output_matrix_width,
                        learn_reduction,
                        cost_deriv_z,
                        layerinfo->filter_feature_output_count,
                        shared_ones_floats,
                        1,
                        0.0f,
                        netstate->weights_f_errors + layerinfo->biases_offset,
                        1
                    );
    // calc cost_deriv_weights
    learn_reduction = 1.0f / (layerinfo->input_matrix_width);
    beta=0.0f;
    for(j=0; j<layerinfo->filter_y_count; j++)
    {
        for(i=0; i<layerinfo->filter_x_count; i++)
        {
            MATH_MULT_MAT_MAT(  CblasRowMajor,
                                CblasTrans,
                                CblasNoTrans,
                                layerinfo->filter_feature_output_count,
                                layerinfo->input_matrix_height,
                                layerinfo->input_matrix_width,
                                learn_reduction,
                                cost_deriv_z
                                 + i * layerinfo->partial_output_matrix_count,
                                layerinfo->filter_feature_output_count,
                                netstate->activations
                                 + layerinfo->input_activation_offset
                                 + i * layerinfo->filter_feature_input_count
                                 + j * layerinfo->input_matrix_toplayer_elements_count,
                                layerinfo->input_matrix_height,
                                beta,
                                netstate->weights_f_errors
                                 + layerinfo->weights_offset
                                 + j * layerinfo->input_matrix_height,
                                layerinfo->filter_matrix_width
                     );
        }
        beta=1.0f;
    }
}


INLINE void layer_conv_first_forward(   const ConvolutionalLayer_p layerinfo,
                                        NetState_p netstate,
                                        const Float_p input_start
                                    )
{
    Float_p z_vector_start = shared_tmp_floats;
    // matrix products are summed up, but cleaned in the beginning
    Float_t beta=0.0f;
    Int_t i, j;
    for(j=0; j<layerinfo->filter_y_count; j++)
    {
        for(i=0; i<layerinfo->filter_x_count; i++)
        {
            MATH_MULT_MAT_MAT(  CblasColMajor,
                                CblasTrans,
                                CblasNoTrans,
                                layerinfo->filter_feature_output_count,
                                layerinfo->input_matrix_width,
                                layerinfo->input_matrix_height,
                                1.0f,
                                netstate->weights_f + layerinfo->weights_offset + j * layerinfo->input_matrix_height,
                                layerinfo->filter_matrix_width,
                                input_start
                                 + layerinfo->input_activation_offset
                                 + i * layerinfo->filter_feature_input_count
                                 + j * layerinfo->input_matrix_toplayer_elements_count,
                                layerinfo->input_matrix_height,
                                beta,
                                z_vector_start + i * layerinfo->partial_output_matrix_count,
                                layerinfo->filter_feature_output_count
                             );
        }
        beta=1.0f;
    }
    // copy and concatenate bias vector; add it to weighted inputs
    MATH_MULT_MAT_MAT(  CblasColMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        layerinfo->filter_feature_output_count,
                        layerinfo->input_matrix_width,
                        1,
                        1.0f,
                        netstate->weights_f + layerinfo->biases_offset,
                        layerinfo->filter_feature_output_count,
                        shared_ones_floats,
                        1,
                        1.0f,
                        z_vector_start,
                        layerinfo->filter_feature_output_count
                    ) ;
    // apply sigmoid to result
    sigmoid(    layerinfo->output_activation_count,
                z_vector_start,
                netstate->activations + layerinfo->output_activation_offset
            ) ;
}


INLINE void layer_conv_first_backward(  const ConvolutionalLayer_p layerinfo,
                                        NetState_p netstate,
                                        const Float_p input_start
                                     )
{
    Float_p y_deriv_z = shared_tmp_floats;
    Float_p rev_sigmoid_buffer = shared_tmp_floats + layerinfo->output_activation_count;
    Float_p cost_deriv_z = rev_sigmoid_buffer;
    Int_t i, j;
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
                        layerinfo->filter_feature_output_count,
                        layerinfo->full_output_matrix_width,
                        1.0f,
                        cost_deriv_z,
                        layerinfo->filter_feature_output_count,
                        shared_ones_floats,
                        1,
                        0.0f,
                        netstate->weights_f_errors + layerinfo->biases_offset,
                        1
                    );
    // calc cost_deriv_weights
    Float_t beta=0.0f;
    for(j=0; j<layerinfo->filter_y_count; j++)
    {
        for(i=0; i<layerinfo->filter_x_count; i++)
        {
            MATH_MULT_MAT_MAT(  CblasRowMajor,
                                CblasTrans,
                                CblasNoTrans,
                                layerinfo->filter_feature_output_count,
                                layerinfo->input_matrix_height,
                                layerinfo->input_matrix_width,
                                1.0f,
                                cost_deriv_z
                                 + i * layerinfo->partial_output_matrix_count,
                                layerinfo->filter_feature_output_count,
                                input_start
                                 + i * layerinfo->filter_feature_input_count
                                 + j * layerinfo->input_matrix_toplayer_elements_count,
                                layerinfo->input_matrix_height,
                                beta,
                                netstate->weights_f_errors
                                 + layerinfo->weights_offset
                                 + j * layerinfo->input_matrix_height,
                                layerinfo->filter_matrix_width
                     );
        }
        beta=1.0f;
    }
}


INLINE Float_p layer_conv_get_output(const ConvolutionalLayer_p layerinfo,
                                     NetState_p netstate
                                    )
{
    return netstate->activations + layerinfo->output_activation_offset;
}
