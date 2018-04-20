#ifndef MKL_WRAPPER_H_INCLUDED
#define MKL_WRAPPER_H_INCLUDED

#include "settings.h"
#include "mkl.h"
#include "mkl_cblas.h"
#include "mkl_vml.h"


#define MATH_MALLOC_F_ARRAY(num_of_floats) ((Float_p)mkl_malloc(num_of_floats*sizeof(Float_t), CONFIG_ARRAY_ALIGNMENT*sizeof(Float_t)))
#define MATH_MALLOC_I_ARRAY(num_of_ints) ((Int_p)mkl_malloc(num_of_ints*sizeof(Int_t), CONFIG_ARRAY_ALIGNMENT*sizeof(Int_t)))
#define MATH_MALLOC_FP_ARRAY(num_of_ints) ((Float_p*)mkl_malloc(num_of_ints*sizeof(Float_p), CONFIG_ARRAY_ALIGNMENT*sizeof(Float_p)))


#ifdef CONFIG_FLOATTYPE_DOUBLE

#define MATH_GET_MAX_INDEX(n, x, incx) cblas_idamax(n, x, incx)
// vector mathematical functions
#define MATH_VECT_ADD_SERIAL(n, a, b, y) vdAdd(n, a, b, y)
#define MATH_VECT_SUB_SERIAL(n, a, b, y) vdSub(n, a, b, y)
#define MATH_VECT_MUL_SERIAL(n, a, b, y) vdMul(n, a, b, y)
#define MATH_VECT_DIV_SERIAL(n, a, b, y) vdMul(n, a, b, y)
#define MATH_VECT_TANH_SERIAL(n, a, y) vdTanh(n, a, y)
#define MATH_VECT_EXP_SERIAL(n, a, y) vdExp(n, a, y)
#define MATH_VECT_LOG_SERIAL(n, a, y) vdLn(n, a, y)
// BLAS lvl 1
#define MATH_VECT_ELEM_SUM(n, x, incx) cblas_dasum (n, x, incx)
#define MATH_VECT_SCAL_MUL(n, a, x, incx) cblas_dscal(n, a, x, incx)
#define MATH_VECT_VECT_SCAL_ADD_MUL(n, a, x, incx, b, y, incy) cblas_daxpby(n, a, x, incx, b, y, incy)
#define MATH_VECT_VECT_SCAL_ADD(n, a, x, incx, y, incy) cblas_daxpy(n, a, x, incx, y, incy)
// BLAS lvl 2
#define MATH_MULT_MAT_VECT(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) cblas_dgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
// BLAS lvl 3
#define MATH_MULT_MAT_MAT(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) cblas_dgemm(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

#else /*CONFIG_FLOATTYPE_DOUBLE*/

#define MATH_GET_MAX_INDEX(n, x, incx) cblas_isamax(n, x, incx)
// vector mathematical functions
#define MATH_VECT_ADD_SERIAL(n, a, b, y) vsAdd(n, a, b, y)
#define MATH_VECT_SUB_SERIAL(n, a, b, y) vsSub(n, a, b, y)
#define MATH_VECT_MUL_SERIAL(n, a, b, y) vsMul(n, a, b, y)
#define MATH_VECT_DIV_SERIAL(n, a, b, y) vsMul(n, a, b, y)
#define MATH_VECT_TANH_SERIAL(n, a, y) vsTanh(n, a, y)
#define MATH_VECT_EXP_SERIAL(n, a, y) vsExp(n, a, y)
#define MATH_VECT_LOG_SERIAL(n, a, y) vsLn(n, a, y)
// BLAS lvl 1
#define MATH_VECT_ELEM_SUM(n, x, incx) cblas_sasum (n, x, incx)
#define MATH_VECT_SCAL_MUL(n, a, x, incx) cblas_sscal(n, a, x, incx)
#define MATH_VECT_VECT_SCAL_ADD_MUL(n, a, x, incx, b, y, incy) cblas_saxpby(n, a, x, incx, b, y, incy)
#define MATH_VECT_VECT_SCAL_ADD(n, a, x, incx, y, incy) cblas_saxpy(n, a, x, incx, y, incy)
// BLAS lvl 2
#define MATH_MULT_MAT_VECT(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) cblas_sgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
// BLAS lvl 3
#define MATH_MULT_MAT_MAT(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) cblas_sgemm(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

#endif /*CONFIG_FLOATTYPE_DOUBLE*/

// vector mathematical functions (parallel)
#ifdef CONFIG_MATH_VECT_PARALLEL
void MATH_VECT_ADD(Int_t n, Float_p a, Float_p b, Float_p y);
void MATH_VECT_SUB(Int_t n, Float_p a, Float_p b, Float_p y);
void MATH_VECT_MUL(Int_t n, Float_p a, Float_p b, Float_p y);
void MATH_VECT_DIV(Int_t n, Float_p a, Float_p b, Float_p y);
void MATH_VECT_TANH(Int_t n, Float_p a, Float_p y);
void MATH_VECT_EXP(Int_t n, Float_p a, Float_p y);
void MATH_VECT_LOG(Int_t n, Float_p a, Float_p y);
#else /* CONFIG_MATH_VECT_PARALLEL */
#define MATH_VECT_ADD   MATH_VECT_ADD_SERIAL
#define MATH_VECT_SUB   MATH_VECT_SUB_SERIAL
#define MATH_VECT_MUL   MATH_VECT_MUL_SERIAL
#define MATH_VECT_DIV   MATH_VECT_DIV_SERIAL
#define MATH_VECT_TANH  MATH_VECT_TANH_SERIAL
#define MATH_VECT_EXP   MATH_VECT_EXP_SERIAL
#define MATH_VECT_LOG   MATH_VECT_LOG_SERIAL
#endif /* CONFIG_MATH_VECT_PARALLEL */


#endif /*MKL_WRAPPER_H_INCLUDED*/
