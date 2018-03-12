#ifndef SETTINGS_H_INCLUDED
#define SETTINGS_H_INCLUDED

#ifdef CONFIG_FLOATTYPE_FLOAT
typedef float Float_t, *Float_p
#endif

#ifdef CONFIG_FLOATTYPE_DOUBLE
typedef double Float_t, *Float_p
#endif

typedef int Int_t, *Int_p

#define INLINE inline

#endif /*SETTINGS_H_INCLUDED*/
