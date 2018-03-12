#include "mathematics.h"
#include "shared_arrays.h"


void sigmoid(Int_t count, Float_p input, Float_p output)
{
    MATH_VECT_SCAL_MUL(count, 0.5f, input, 1);
    MATH_VECT_TANH(count, input, output);
    MATH_VECT_VECT_SCAL_ADD_MUL(count, 0.5f, shared_ones_floats, 1, 0.5f, output, 1);
}

void sigmoid_derivation(Int_t count, Float_p activation, Float_p derivation, Float_p tmp)
{
    MATH_VECT_SUB(count, shared_ones_floats, activation, tmp);
    MATH_VECT_MUL(count, tmp, activation, derivation);
}
