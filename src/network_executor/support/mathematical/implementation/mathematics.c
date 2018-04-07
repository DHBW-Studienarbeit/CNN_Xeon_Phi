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
    Float_t sum;
    for(i=0; i<count_probes; i++)
    {
        MATH_VECT_EXP(probe_size, output, temporary+i*probe_size);
        sum = MATH_VECT_ELEM_SUM(probe_size, temporary+i*probe_size, 1);
        MATH_VECT_SCAL_MUL(probe_size, (1.0f/sum), temporary+i*probe_size, 1);
    }
    // calculate cost by cross entropy
    sum=0.0f;
    MATH_VECT_LOG(count, temporary, temp2);
    MATH_VECT_MUL(count, temp2, labels, temporary);
    sum = MATH_VECT_ELEM_SUM(count, temporary, 1);
    return (-1.0f) * sum;
}

INLINE void get_cost_derivatives(Int_t count, Float_p output, Float_p labels, Float_p derivatives)
{
    MATH_VECT_SUB(count, output, labels, derivatives);
}
