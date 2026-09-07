#ifndef PYADH_LAPACK_PROTO_H
#define PYADH_LAPACK_PROTO_H

#ifdef __cplusplus
extern "C"
{
#endif

/* These are all Fortran SUBROUTINEs, so they return nothing -- they report
 * status through their trailing `info` argument, and no caller in proteus
 * looks at a return value. They were declared `int` here, which is what
 * classic f2c emits for a subroutine, but not what a Fortran-compiled
 * LAPACK provides, nor what PETSc's f2cblaslapack provides (its
 * lapack/dgetrf.c defines `void dgetrf_(...)`).
 *
 * On ELF the mismatch is invisible: the caller simply ignores whatever is
 * in the return register. Not so on emscripten-wasm32, where a function's
 * type is part of its identity and enforced when the call is made, so
 * calling a `-> void` definition through an `-> i32` declaration traps the
 * VM outright. That showed up as a bare `RuntimeError: unreachable` inside
 * cfemIntegrals' estimate_mt_lowmem(), the first thing in a proteus solve
 * to call dgetrf_/dgetrs_, with wasm-ld having warned at link time:
 *   function signature mismatch: dgetrf_
 *   >>> defined as (i32, i32, i32, i32, i32, i32) -> i32  in smoothers.o
 *   >>> defined as (i32, i32, i32, i32, i32, i32) -> void in libf2clapack.a
 */
extern void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern void dgetc2_(int *n, double *a, int *lda, int *ipiv, int *jpiv, int *info);
extern void dgesc2_(int *n, double *a, int *lda, double* rhs, int *ipiv, int *jpiv, double* scale);
extern void dgeev_(char* jobvl, char* jobvr, int* n, double* a, int* lda, double* wr, double* wi, double* vl, int* ldvl, double* vr, int* ldvr, double* work, int* lwork,int* info);
extern void dgetri_(int* N,double* A,int* LDA,int* IPIV,double* WORK,int* LWORK,int* INFO );

#ifdef __cplusplus
}
#endif

#define __CLPK_integer int
#endif
