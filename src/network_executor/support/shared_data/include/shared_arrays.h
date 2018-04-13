#ifndef SHARED_ARRAYS_H_INCLUDED
#define SHARED_ARRAYS_H_INCLUDED

#include "settings.h"


/* [[[cog
import cog
from network_descriptor.NetInstance import net

cog.outl("#define SHARED_ARRAY_SIZE " + str(net._max_act_count))
cog.outl("#define SHARED_TMP_ARRAY_SIZE " + str(2 * net._max_act_count))
]]] */
#define SHARED_ARRAY_SIZE 92700
#define SHARED_TMP_ARRAY_SIZE 185400
// [[[end]]]

extern Float_t shared_ones_floats[SHARED_ARRAY_SIZE];

extern Float_t shared_tmp_floats[SHARED_TMP_ARRAY_SIZE];
extern Int_t shared_tmp_ints[SHARED_TMP_ARRAY_SIZE];


#endif /*SHARED_ARRAYS_H_INCLUDED*/
