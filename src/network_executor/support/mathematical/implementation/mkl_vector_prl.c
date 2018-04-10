#include "settings.h"
#include "mkl_wrapper.h"


#ifdef CONFIG_MATH_VECT_PARALLEL

void MATH_VECT_ADD(Int_t n, Float_p a, Float_p b, Float_p y)
{
    Int_t i;
    Int_t stdsize = n / CONFIC_VECT_NUM_THREADS;
    Int_t lastsize = stdsize + n % CONFIC_VECT_NUM_THREADS;
    #pragma omp parallel for
    for(i=0; i<CONFIC_VECT_NUM_THREADS; i++)
    {
        Int_t thread_offset = i * stdsize;
        Int_t thread_size = (i==CONFIC_VECT_NUM_THREADS-1)?lastsize:stdsize;
        MATH_VECT_ADD_SERIAL(thread_size, a+thread_offset, b+thread_offset, y+thread_offset);
    }
}

void MATH_VECT_SUB(Int_t n, Float_p a, Float_p b, Float_p y)
{
    Int_t i;
    Int_t stdsize = n / CONFIC_VECT_NUM_THREADS;
    Int_t lastsize = stdsize + n % CONFIC_VECT_NUM_THREADS;
    #pragma omp parallel for
    for(i=0; i<CONFIC_VECT_NUM_THREADS; i++)
    {
        Int_t thread_offset = i * stdsize;
        Int_t thread_size = (i==CONFIC_VECT_NUM_THREADS-1)?lastsize:stdsize;
        MATH_VECT_SUB_SERIAL(thread_size, a+thread_offset, b+thread_offset, y+thread_offset);
    }
}

void MATH_VECT_MUL(Int_t n, Float_p a, Float_p b, Float_p y)
{
    Int_t i;
    Int_t stdsize = n / CONFIC_VECT_NUM_THREADS;
    Int_t lastsize = stdsize + n % CONFIC_VECT_NUM_THREADS;
    #pragma omp parallel for
    for(i=0; i<CONFIC_VECT_NUM_THREADS; i++)
    {
        Int_t thread_offset = i * stdsize;
        Int_t thread_size = (i==CONFIC_VECT_NUM_THREADS-1)?lastsize:stdsize;
        MATH_VECT_MUL_SERIAL(thread_size, a+thread_offset, b+thread_offset, y+thread_offset);
    }
}

void MATH_VECT_DIV(Int_t n, Float_p a, Float_p b, Float_p y)
{
    Int_t i;
    Int_t stdsize = n / CONFIC_VECT_NUM_THREADS;
    Int_t lastsize = stdsize + n % CONFIC_VECT_NUM_THREADS;
    #pragma omp parallel for
    for(i=0; i<CONFIC_VECT_NUM_THREADS; i++)
    {
        Int_t thread_offset = i * stdsize;
        Int_t thread_size = (i==CONFIC_VECT_NUM_THREADS-1)?lastsize:stdsize;
        MATH_VECT_DIV_SERIAL(thread_size, a+thread_offset, b+thread_offset, y+thread_offset);
    }
}

void MATH_VECT_TANH(Int_t n, Float_p a, Float_p y)
{
    Int_t i;
    Int_t stdsize = n / CONFIC_VECT_NUM_THREADS;
    Int_t lastsize = stdsize + n % CONFIC_VECT_NUM_THREADS;
    #pragma omp parallel for
    for(i=0; i<CONFIC_VECT_NUM_THREADS; i++)
    {
        Int_t thread_offset = i * stdsize;
        Int_t thread_size = (i==CONFIC_VECT_NUM_THREADS-1)?lastsize:stdsize;
        MATH_VECT_TANH_SERIAL(thread_size, a+thread_offset, y+thread_offset);
    }
}

void MATH_VECT_EXP(Int_t n, Float_p a, Float_p y)
{
    Int_t i;
    Int_t stdsize = n / CONFIC_VECT_NUM_THREADS;
    Int_t lastsize = stdsize + n % CONFIC_VECT_NUM_THREADS;
    #pragma omp parallel for
    for(i=0; i<CONFIC_VECT_NUM_THREADS; i++)
    {
        Int_t thread_offset = i * stdsize;
        Int_t thread_size = (i==CONFIC_VECT_NUM_THREADS-1)?lastsize:stdsize;
        MATH_VECT_EXP_SERIAL(thread_size, a+thread_offset, y+thread_offset);
    }
}

void MATH_VECT_LOG(Int_t n, Float_p a, Float_p y)
{
    Int_t i;
    Int_t stdsize = n / CONFIC_VECT_NUM_THREADS;
    Int_t lastsize = stdsize + n % CONFIC_VECT_NUM_THREADS;
    #pragma omp parallel for
    for(i=0; i<CONFIC_VECT_NUM_THREADS; i++)
    {
        Int_t thread_offset = i * stdsize;
        Int_t thread_size = (i==CONFIC_VECT_NUM_THREADS-1)?lastsize:stdsize;
        MATH_VECT_LOG_SERIAL(thread_size, a+thread_offset, y+thread_offset);
    }
}

#endif /* CONFIG_MATH_VECT_PARALLEL */
