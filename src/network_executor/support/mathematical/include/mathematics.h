#ifndef MATHEMATICS_H_INCLUDED
#define MATHEMATICS_H_INCLUDED

#include "settings.h"
#include "mkl_wrapper.h"


INLINE void sigmoid(Int_t count, Float_p input, Float_p output);
INLINE void sigmoid_derivation(Int_t count, Float_p activation, Float_p derivation, Float_p tmp);

INLINE Float_t get_cost(Int_t count_probes, Int_t probe_size, Float_p output, Float_p labels, Float_p temporary);
INLINE Float_t get_accuracy(Int_t count_probes, Int_t probe_size, Float_p output, Float_p labels);
INLINE void get_cost_derivatives(Int_t count, Float_p output, Float_p labels, Float_p derivatives);

#endif /*MATHEMATICS_H_INCLUDED*/
