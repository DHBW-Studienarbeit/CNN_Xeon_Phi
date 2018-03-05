#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "settings.h"

// depends on network description
#define NETWORKDATA_SIZE  1000;

typedef struct {
    Float_t network_data[NETWORKDATA_SIZE];
} Network_t, *Network_p;



#endif /*NETWORK_H_INCLUDED*/
