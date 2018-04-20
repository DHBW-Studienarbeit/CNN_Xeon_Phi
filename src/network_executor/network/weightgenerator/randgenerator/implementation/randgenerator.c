#include "weightgenerator.h"
#include "mkl_vsl.h"
#include "config.h"


#define RANDGEN_GENMODE     VSL_BRNG_SFMT19937
#define RANDGEN_DISTMODE    VSL_RNG_METHOD_GAUSSIAN_ICDF



void weightgen_generate(Int_t count, Float_p target)
{
    VSLStreamStatePtr stream;
    vslNewStream( &stream, RANDGEN_GENMODE, CONFIG_RANDGEN_SEED );
#ifdef CONFIG_FLOATTYPE_DOUBLE
    vdRngGaussian( VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, count, target, CONFIG_RANDGEN_MEAN, CONFIG_RANDGEN_RANGESIZE );
#else
    vsRngGaussian( VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, count, target, CONFIG_RANDGEN_MEAN, CONFIG_RANDGEN_RANGESIZE );
#endif
    vslDeleteStream( &stream );
}
