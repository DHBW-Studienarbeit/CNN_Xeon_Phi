#ifndef MATHEMATICS_H_INCLUDED
#define MATHEMATICS_H_INCLUDED

#include "settings.h"
#include "mkl_wrapper.h"


void sigmoid(Int_t count, Float_p input, Float_p output);
void sigmoid_derivation(Int_t count, Float_p activation, Float_p derivation, Float_p tmp);

Float_t get_cost(Float_p output, Float_p labels, Float_p temporary, Int_t size);
void get_cost_derivatives(Float_p output, Float_p labels, Float_p derivatives, Int_t size);

#endif /*MATHEMATICS_H_INCLUDED*/
