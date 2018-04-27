#include "convlayer.h"
#include "mathematics.h"


INLINE void layer_conv_forward( const ConvolutionalLayer_p layerinfo)
{
    Float_p z_vector_start = layerinfo->shared_tmp_floats;
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
                                layerinfo->weights_start + j * layerinfo->input_matrix_height,
                                layerinfo->filter_matrix_width,
                                layerinfo->input_activation_start
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
                        layerinfo->biases_start,
                        layerinfo->filter_feature_output_count,
                        layerinfo->shared_ones_floats,
                        1,
                        1.0f,
                        z_vector_start,
                        layerinfo->filter_feature_output_count
                    ) ;
    // apply sigmoid to result
    sigmoid(    layerinfo->output_activation_count,
                z_vector_start,
                layerinfo->output_activation_start
            ) ;
}

INLINE void layer_conv_backward(const ConvolutionalLayer_p layerinfo)
{
    Float_p y_deriv_z = layerinfo->shared_tmp_floats;
    Float_p rev_sigmoid_buffer = y_deriv_z + layerinfo->output_activation_count;
    Float_p cost_deriv_z = rev_sigmoid_buffer;
    Int_t i, j;
    // calc y_deriv_z from y
    sigmoid_derivation( layerinfo->output_activation_count,
                        layerinfo->output_activation_start,
                        y_deriv_z,
                        rev_sigmoid_buffer
                      );
    // calc cost_deriv_z from y_deriv_z and cost_deriv_y
    MATH_VECT_MUL(  layerinfo->output_activation_count,
                    y_deriv_z,
                    layerinfo->output_activation_error_start,
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
                            layerinfo->weights_start + j * layerinfo->input_matrix_height,
                            layerinfo->filter_matrix_width,
                            cost_deriv_z + i * layerinfo->partial_output_matrix_count,
                            layerinfo->filter_feature_output_count,
                            beta,
                            layerinfo->input_activation_error_start
                            + i * layerinfo->filter_feature_input_count
                            + j * layerinfo->input_matrix_toplayer_elements_count,
                            layerinfo->input_matrix_height
                      );
        }
        beta=1.0f;
    }

    // add cost_deriv_z to cost_deriv_bias for all datasets of the batch
    MATH_MULT_MAT_VECT( CblasColMajor,
                        CblasNoTrans,
                        layerinfo->filter_feature_output_count,
                        layerinfo->full_output_matrix_width,
                        layerinfo->learn_reduction_biases,
                        cost_deriv_z,
                        layerinfo->filter_feature_output_count,
                        layerinfo->shared_ones_floats,
                        1,
                        0.0f,
                        layerinfo->biases_error_start,
                        1
                    );
    // calc cost_deriv_weights
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
                                layerinfo->learn_reduction_weights,
                                cost_deriv_z
                                 + i * layerinfo->partial_output_matrix_count,
                                layerinfo->filter_feature_output_count,
                                layerinfo->input_activation_start
                                 + i * layerinfo->filter_feature_input_count
                                 + j * layerinfo->input_matrix_toplayer_elements_count,
                                layerinfo->input_matrix_height,
                                beta,
                                layerinfo->weights_error_start
                                 + j * layerinfo->input_matrix_height,
                                layerinfo->filter_matrix_width
                     );
        }
        beta=1.0f;
    }
}


INLINE void layer_conv_first_forward(   const ConvolutionalLayer_p layerinfo,
                                        const Float_p input_start
                                    )
{
    Float_p z_vector_start = layerinfo->shared_tmp_floats;
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
                                layerinfo->weights_start + j * layerinfo->input_matrix_height,
                                layerinfo->filter_matrix_width,
                                input_start
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
                        layerinfo->biases_start,
                        layerinfo->filter_feature_output_count,
                        layerinfo->shared_ones_floats,
                        1,
                        1.0f,
                        z_vector_start,
                        layerinfo->filter_feature_output_count
                    ) ;
    // apply sigmoid to result
    sigmoid(    layerinfo->output_activation_count,
                z_vector_start,
                layerinfo->output_activation_start
            ) ;
}


INLINE void layer_conv_first_backward(  const ConvolutionalLayer_p layerinfo,
                                        const Float_p input_start
                                     )
{
    Float_p y_deriv_z = layerinfo->shared_tmp_floats;
    Float_p rev_sigmoid_buffer = y_deriv_z + layerinfo->output_activation_count;
    Float_p cost_deriv_z = rev_sigmoid_buffer;
    Int_t i, j;
    // calc y_deriv_z from y
    sigmoid_derivation( layerinfo->output_activation_count,
                        layerinfo->output_activation_start,
                        y_deriv_z,
                        rev_sigmoid_buffer
                      );
    // calc cost_deriv_z from y_deriv_z and cost_deriv_y
    MATH_VECT_MUL(  layerinfo->output_activation_count,
                    y_deriv_z,
                    layerinfo->output_activation_error_start,
                    cost_deriv_z
                 );

    // add cost_deriv_z to cost_deriv_bias for all datasets of the batch
    MATH_MULT_MAT_VECT( CblasColMajor,
                        CblasNoTrans,
                        layerinfo->filter_feature_output_count,
                        layerinfo->full_output_matrix_width,
                        layerinfo->learn_reduction_biases,
                        cost_deriv_z,
                        layerinfo->filter_feature_output_count,
                        layerinfo->shared_ones_floats,
                        1,
                        0.0f,
                        layerinfo->biases_error_start,
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
                                layerinfo->learn_reduction_weights,
                                cost_deriv_z
                                 + i * layerinfo->partial_output_matrix_count,
                                layerinfo->filter_feature_output_count,
                                input_start
                                 + i * layerinfo->filter_feature_input_count
                                 + j * layerinfo->input_matrix_toplayer_elements_count,
                                layerinfo->input_matrix_height,
                                beta,
                                layerinfo->weights_error_start
                                 + j * layerinfo->input_matrix_height,
                                layerinfo->filter_matrix_width
                     );
        }
        beta=1.0f;
    }
}


INLINE Float_p layer_conv_get_output(const ConvolutionalLayer_p layerinfo)
{
    return layerinfo->output_activation_start;
}

INLINE Int_t layer_conv_get_output_position(const ConvolutionalLayer_p layerinfo,
                                            Int_t p,
                                            Int_t y,
                                            Int_t x,
                                            Int_t f
                                           )
{
    Int_t input_start_position = x + y*layerinfo->input_x_count + p*layerinfo->input_xy_count;
    Int_t output_section_index = input_start_position % layerinfo->filter_x_count;
    Int_t inner_section_column = input_start_position / layerinfo->filter_x_count;
    Int_t output_section_size = layerinfo->input_matrix_width * layerinfo->filter_feature_output_count;
    return output_section_index * output_section_size + inner_section_column * layerinfo->filter_feature_output_count + f;
}
