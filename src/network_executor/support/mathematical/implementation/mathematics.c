#include "mathematics.h"
#include "shared_arrays.h"


INLINE void sigmoid(Int_t count, Float_p input, Float_p output)
{
    MATH_VECT_SCAL_MUL(count, 0.5f, input, 1);
    MATH_VECT_TANH(count, input, output);
    MATH_VECT_VECT_SCAL_ADD_MUL(count, 0.5f, shared_ones_floats, 1, 0.5f, output, 1);
}

INLINE void sigmoid_derivation(Int_t count, Float_p activation, Float_p derivation, Float_p tmp)
{
    MATH_VECT_SUB(count, shared_ones_floats, activation, tmp);
    MATH_VECT_MUL(count, tmp, activation, derivation);
}

INLINE Float_t get_cost(Int_t count_probes, Int_t probe_size, Float_p output, Float_p labels, Float_p temporary)
{
    Int_t count = count_probes * probe_size;
    Int_t i;
    Float_p temp2 = temporary + count;
    // do softmax first, scaled vector will be stored in temporary
    #pragma omp parallel for
    for(i=0; i<count_probes; i++)
    {
        Float_t sum;
        MATH_VECT_EXP_SERIAL(probe_size, output+i*probe_size, temporary+i*probe_size);
        sum = MATH_VECT_ELEM_SUM(probe_size, temporary+i*probe_size, 1);
        MATH_VECT_SCAL_MUL(probe_size, (1.0f/sum), temporary+i*probe_size, 1);
    }
    // calculate cost by cross entropy
    MATH_VECT_LOG(count, temporary, temp2);
    MATH_VECT_MUL(count, temp2, labels, temporary);
    return MATH_VECT_ELEM_SUM(count, temporary, 1);
}

INLINE Float_t get_accuracy(Int_t count_probes, Int_t probe_size, Float_p output, Float_p labels)
{
    Int_t i, sum=0, prediction, desired;
    for(i=0; i<count_probes; i++)
    {
        prediction = MATH_GET_MAX_INDEX(probe_size, output + probe_size*i, 1);
        desired = MATH_GET_MAX_INDEX(probe_size, labels + probe_size*i, 1);
        if(prediction==desired)
        {
            sum++;
        }
    }
    return ((Float_t)sum)/((Float_t)count_probes);
}

INLINE void get_cost_derivatives(Int_t count_probes, Int_t probe_size, Float_p output, Float_p labels, Float_p derivatives, Float_p temporary)
{
    Int_t i;
    Int_t count = count_probes * probe_size;
    // do softmax first, scaled vector will be stored in temporary
    #pragma omp parallel for
    for(i=0; i<count_probes; i++)
    {
        Float_t sum;
        MATH_VECT_EXP_SERIAL(probe_size, output+i*probe_size, temporary+i*probe_size);
        sum = MATH_VECT_ELEM_SUM(probe_size, temporary+i*probe_size, 1);
        MATH_VECT_SCAL_MUL(probe_size, (1.0f/sum), temporary+i*probe_size, 1);
    }
    // claculated output - desired output
    MATH_VECT_SUB(count, temporary, labels, derivatives);
}
