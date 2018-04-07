#ifndef MKL_WRAPPER_H_INCLUDED
#define MKL_WRAPPER_H_INCLUDED

#include "settings.h"
#include "mkl_cblas.h"
#include "mkl_vml.h"


#ifdef CONFIG_FLOATTYPE_DOUBLE

// vector mathematical functions
#define MATH_VECT_ADD(n, a, b, y) vdAdd(n, a, b, y)
#define MATH_VECT_SUB(n, a, b, y) vdSub(n, a, b, y)
#define MATH_VECT_MUL(n, a, b, y) vdMul(n, a, b, y)
#define MATH_VECT_DIV(n, a, b, y) vdMul(n, a, b, y)
#define MATH_VECT_TANH(n, a, y) vdTanh(n, a, y)
#define MATH_VECT_EXP(n, a, y) vdExp(n, a, y)
#define MATH_VECT_LOG(n, a, y) vdLn(n, a, y)

// BLAS lvl 1
#define MATH_VECT_ELEM_SUM(n, x, incx) cblas_dasum (n, x, incx)
#define MATH_VECT_SCAL_MUL(n, a, x, incx) cblas_dscal(n, a, x, incx)
#define MATH_VECT_VECT_SCAL_ADD_MUL(n, a, x, incx, b, y, incy) cblas_daxpby(n, a, x, incx, b, y, incy)
#define MATH_VECT_VECT_SCAL_ADD(n, a, x, incx, y, incy) cblas_daxpy(n, a, x, incx, y, incy)

// BLAS lvl 2
#define MATH_MULT_MAT_VECT(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) cblas_dgemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

// BLAS lvl 3
#define MATH_MULT_MAT_MAT(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) cblas_dgemm(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

//mBLAS extensions
#define MATH_BATCH_MULT_MAT_MAT(layout, transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size) cblas_dgemm_batch(layout, transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size)


#else /*CONFIG_FLOATTYPE_DOUBLE*/

#define MATH_MULT_MAT_VECT cblas_fgmv
#define MATH_MULT_MAT_MAT cblas_fgemm



#endif /*CONFIG_FLOATTYPE_DOUBLE*/


#endif /*MKL_WRAPPER_H_INCLUDED*/
