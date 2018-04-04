#include "weightgenerator.h"
#include "mkl_vsl.h"

#define RANDGEN_SEED        777
#define RANDGEN_GENMODE     VSL_BRNG_SFMT19937
#define RANDGEN_DISTMODE    VSL_RNG_METHOD_GAUSSIAN_ICDF
#define RANDGEN_MEAN        0.0f
#define RANDGEN_RANGESIZE   5.0f

void weightgen_generate(Int_t count, Float_p target)
{
    VSLStreamStatePtr stream;
    vslNewStream( &stream, RANDGEN_GENMODE, RANDGEN_SEED );
    vdRngGaussian( VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, count, target, RANDGEN_MEAN, RANDGEN_RANGESIZE );
    vslDeleteStream( &stream );
}
