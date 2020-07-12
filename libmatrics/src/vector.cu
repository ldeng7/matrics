#include "internal.h"

// ind

const uint32 VEC_IND_NTB = 512;

#define devVecInd(exp) \
    uint32 i = blockIdx.x * VEC_IND_NTB + threadIdx.x; \
    if (i < n) { exp; }

#define mtVecInd(devFunc, ...) \
    cudaGetLastError(); \
    devFunc<<<divCeil(pX->nx, VEC_IND_NTB), VEC_IND_NTB, 0, stream>>>( \
        pX->p, pX->nx, __VA_ARGS__); \
    syncCheck(stream, pCode);

__global__ void devVecAddScalar(float *x, uint32 n, float *y, float a) {
    devVecInd(y[i] = x[i] + a);
}

__export void mtVecAddScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecAddScalar, pY->p, *pA);
}

__global__ void devVecMulScalar(float *x, uint32 n, float *y, float a) {
    devVecInd(y[i] = a * x[i]);
}

__export void mtVecMulScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecMulScalar, pY->p, *pA);
}

__global__ void devVecMulAddScalar(float *x, uint32 n, float *y, float a, float b) {
    devVecInd(y[i] = a * x[i] + b);
}

__export void mtVecMulAddScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecMulAddScalar, pY->p, *pA, *pB);
}

__global__ void devVecPowScalar(float *x, uint32 n, float *y, float a) {
    devVecInd(y[i] = __powf(x[i], a));
}

__export void mtVecPowScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecPowScalar, pY->p, *pA);
}

__global__ void devVecPowMulScalar(float *x, uint32 n, float *y, float a, float b) {
    devVecInd(y[i] = __powf(x[i], a) * b);
}

__export void mtVecPowMulScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecPowMulScalar, pY->p, *pA, *pB);
}

__global__ void devVecAddVec(float *x, uint32 n, float *y, float *z) {
    devVecInd(z[i] = x[i] + y[i]);
}

__export void mtVecAddVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecAddVec, pY->p, pZ->p);
}

__global__ void devVecSubVec(float *x, uint32 n, float *y, float *z) {
    devVecInd(z[i] = x[i] - y[i]);
}

__export void mtVecSubVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecSubVec, pY->p, pZ->p);
}

__global__ void devVecPatchMulVec(float *x, uint32 n, float *y, float *z) {
    devVecInd(z[i] = x[i] * y[i]);
}

__export void mtVecPatchMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecPatchMulVec, pY->p, pZ->p);
}

// acc

const uint32 VEC_ACC_NTB = 512;
const uint32 VEC_ACC_NPT = 8;

__export void mtNewVecAccBuffer(MtTensor *pTen, buffer *pBuf, int *pCode) {
    newBuffer(divCeil(pTen->nx, VEC_ACC_NTB * VEC_ACC_NPT), pBuf, pCode);
}

#define devVecAcc(rtInit, rbExp, ...) \
    uint32 itb = threadIdx.x; \
    uint32 i = blockIdx.x * VEC_ACC_NTB * VEC_ACC_NPT + itb; \
    float rt = (rtInit); \
    for (uint32 j = 0; j < VEC_ACC_NPT && i < n; j++, i += VEC_ACC_NTB) { __VA_ARGS__; } \
    __shared__ float rb[VEC_ACC_NTB]; \
    rb[itb] = rt; \
    __syncthreads(); \
    for (uint32 i = VEC_ACC_NTB >> 1; i != 0; i >>= 1) { \
        if (itb < i) { rbExp; } \
        __syncthreads(); \
    } \
    if (0 == itb) r[blockIdx.x] = rb[0];

#define mtVecAcc(devFunc, rInit, rExp, ...) \
    float *rg = (float *)buf; \
    uint32 nb = divCeil(pX->nx, VEC_ACC_NTB * VEC_ACC_NPT); \
    cudaGetLastError(); \
    devFunc<<<nb, VEC_ACC_NTB, 0, stream>>>(pX->p, pX->nx, __VA_ARGS__); \
    syncCheck(stream, pCode); \
    float r = (rInit); \
    for (uint32 i = 0; i < nb; i++) { rExp; } \
    *pRes = r;

__global__ void devVecSum(float *x, uint32 n, float *r) {
    devVecAcc(0, rb[itb] += rb[itb + i], rt += x[i]);
}

__export void mtVecSum(MtTensor *pX, buffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecSum, 0, r += rg[i], rg);
}

__global__ void devVecSquareSum(float *x, uint32 n, float *r) {
    float e;
    devVecAcc(0, rb[itb] += rb[itb + i], e = x[i], rt += e * e);
}

__export void mtVecSquareSum(MtTensor *pX, buffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecSquareSum, 0, r += rg[i], rg);
}

__global__ void devVecMin(float *x, uint32 n, float *r) {
    devVecAcc(NAN, rb[itb] = fminf(rb[itb], rb[itb + i]), rt = fminf(x[i], rt));
}

__export void mtVecMin(MtTensor *pX, buffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecMin, NAN, r = fminf(rg[i], r), rg);
}

__global__ void devVecMax(float *x, uint32 n, float *r) {
    devVecAcc(NAN, rb[itb] = fmaxf(rb[itb], rb[itb + i]), rt = fmaxf(x[i], rt));
}

__export void mtVecMax(MtTensor *pX, buffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecMax, NAN, r = fmaxf(rg[i], r), rg);
}

__global__ void devVecDot(float *x, uint32 n, float *y, float *r) {
    devVecAcc(0, rb[itb] += rb[itb + i], rt += x[i] * y[i]);
}

__export void mtVecDot(MtTensor *pX, MtTensor *pY, buffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecDot, 0, r += rg[i], pY->p, rg);
}

__global__ void devVecSumSquareSum(float *x, uint32 n, float *y, float *r) {
    float e;
    devVecAcc(0, rb[itb] += rb[itb + i], e = x[i] + y[i], rt += e * e);
}

__export void mtVecSumSquareSum(MtTensor *pX, MtTensor *pY, buffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecSumSquareSum, 0, r += rg[i], pY->p, rg);
}

__global__ void devVecDiffSquareSum(float *x, uint32 n, float *y, float *r) {
    float e;
    devVecAcc(0, rb[itb] += rb[itb + i], e = x[i] - y[i], rt += e * e);
}

__export void mtVecDiffSquareSum(MtTensor *pX, MtTensor *pY, buffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecDiffSquareSum, 0, r += rg[i], pY->p, rg);
}
