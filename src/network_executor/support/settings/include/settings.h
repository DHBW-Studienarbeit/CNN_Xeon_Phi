#ifndef SETTINGS_H_INCLUDED
#define SETTINGS_H_INCLUDED

#include "config.h"



#ifdef CONFIG_FLOATTYPE_FLOAT
typedef float Float_t, *Float_p;
#define FLOAT_T_ESCAPE "%.5f"
#endif

#ifdef CONFIG_FLOATTYPE_DOUBLE
typedef double Float_t, *Float_p;
#define FLOAT_T_ESCAPE "%.5lf"
#endif

typedef int Int_t, *Int_p;

//#define INLINE inline
#define INLINE



#include "testing.h"

#endif /*SETTINGS_H_INCLUDED*/
