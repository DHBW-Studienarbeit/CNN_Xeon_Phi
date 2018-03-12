#ifndef SHARED_ARRAYS_H_INCLUDED
#define SHARED_ARRAYS_H_INCLUDED

#include "settings.h"

#define SHARED_ARRAY_SIZE 1024
#define SHARED_TMP_ARRAY_SIZE 2048

Float_t shared_ones_floats[SHARED_ARRAY_SIZE];
FLoat_t shared_halfs_floats[SHARED_ARRAY_SIZE];


Int_t   shared_count_ints[SHARED_ARRAY_SIZE];
Int_t   shared_count_step4_ints[SHARED_ARRAY_SIZE];


Float_t shared_tmp_floats[SHARED_TMP_ARRAY_SIZE];
Int_t shared_tmp_ints[SHARED_TMP_ARRAY_SIZE];


#endif /*SHARED_ARRAYS_H_INCLUDED*/
