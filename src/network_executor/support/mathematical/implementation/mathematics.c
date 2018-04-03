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

Float_t get_cost(Int_t count, Float_p output, Float_p labels, Float_p temporary)
{
    Float_p temp2 = temporary + count;
    // do softmax first, scaled vector will be stored in temporary
    Float_t sum;
    MATH_VECT_EXP(count, output, temporary);
    sum = MATH_VECT_ELEM_SUM(count, temporary, 1);
    MATH_VECT_SCAL_MUL(count, (1.0f/sum), temporary, 1);
    // calculate cost by cross entropy
    sum=0.0f;
    MATH_VECT_LOG(count, temporary, temp2);
    MATH_VECT_MUL(count, temp2, labels, temporary);
    sum = MATH_VECT_ELEM_SUM(count, temporary, 1);
    return (-1.0f) * sum;
}

void get_cost_derivatives(Int_t count, Float_p output, Float_p labels, Float_p derivatives)
{
    MATH_VECT_SUB(count, output, labels, derivatives);
}
