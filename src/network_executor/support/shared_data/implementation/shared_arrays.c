#include "shared_arrays.h"

Float_t shared_ones_floats[SHARED_ARRAY_SIZE] = { 1.0f };

Float_t shared_tmp_floats[SHARED_TMP_ARRAY_SIZE];
Int_t shared_tmp_ints[SHARED_TMP_ARRAY_SIZE];



void init_shared_arrays(void)
{
    Int_t i;
    for(i=0; i<SHARED_ARRAY_SIZE; i++)
    {
        shared_ones_floats[i] = 1.0f;
    }
}
