#include "internal.h"

// ind

const size_t VEC_IND_NTB = 512;

#define devVecInd(...) \
    size_t i = size_t(blockIdx.x) * VEC_IND_NTB + size_t(threadIdx.x); \
    if (i < n) { ##__VA_ARGS__; }

#define mtVecInd(devFunc, ...) \
    cudaGetLastError(); \
    devFunc<<<divCeil(pX->nx, VEC_IND_NTB), VEC_IND_NTB, 0, stream>>>( \
        pX->p, pY->p, ##__VA_ARGS__, pX->nx); \
    syncCheck;

__global__ void devVecAddScalar(float *x, float *y, float a, size_t n) {
    devVecInd(y[i] = x[i] + a);
}

__export void mtVecAddScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecAddScalar, *pA);
}

__global__ void devVecMulScalar(float *x, float *y, float a, size_t n) {
    devVecInd(y[i] = a * x[i]);
}

__export void mtVecMulScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecMulScalar, *pA);
}

__global__ void devVecMulAddScalar(float *x, float *y, float a, float b, size_t n) {
    devVecInd(y[i] = a * x[i] + b);
}

__export void mtVecMulAddScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecMulAddScalar, *pA, *pB);
}

__global__ void devVecPowScalar(float *x, float *y, float a, size_t n) {
    devVecInd(y[i] = __powf(x[i], a));
}

__export void mtVecPowScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecPowScalar, *pA);
}

__global__ void devVecPowMulScalar(float *x, float *y, float a, float b, size_t n) {
    devVecInd(y[i] = __powf(x[i], a) * b);
}

__export void mtVecPowMulScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecPowMulScalar, *pA, *pB);
}

__global__ void devVecAddVec(float *x, float *y, float *z, size_t n) {
    devVecInd(z[i] = x[i] + y[i]);
}

__export void mtVecAddVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecAddVec, pZ->p);
}

__global__ void devVecSubVec(float *x, float *y, float *z, size_t n) {
    devVecInd(z[i] = x[i] - y[i]);
}

__export void mtVecSubVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecSubVec, pZ->p);
}

__global__ void devVecPatchMulVec(float *x, float *y, float *z, size_t n) {
    devVecInd(z[i] = x[i] * y[i]);
}

__export void mtVecPatchMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtVecInd(devVecPatchMulVec, pZ->p);
}

// acc

const size_t VEC_ACC_NTB = 512;
const size_t VEC_ACC_NPT = 8;

__export void mtNewVecAccBuffer(MtTensor *pTen, MtBuffer *pBuf, int *pCode) {
    *pBuf = nullptr;
    MtBuffer buf = mtNewBuffer(divCeil(pTen->nx, VEC_ACC_NTB * VEC_ACC_NPT), pCode);
    if (cudaSuccess != *pCode) return;
    *pBuf = buf;
}

#define devVecAcc(rtInit, rbExp, ...) \
    size_t itb = size_t(threadIdx.x); \
    size_t i = size_t(blockIdx.x) * VEC_ACC_NTB * VEC_ACC_NPT + itb; \
    float rt = (rtInit); \
    for (size_t j = 0; j < VEC_ACC_NPT && i < n; j++, i += VEC_ACC_NTB) { ##__VA_ARGS__; } \
    __shared__ float rb[VEC_ACC_NTB]; \
    rb[itb] = rt; \
    __syncthreads(); \
    for (size_t i = VEC_ACC_NTB >> 1; i != 0; i >>= 1) { \
        if (itb < i) { rbExp; } \
        __syncthreads(); \
    } \
    if (0 == itb) r[blockIdx.x] = rb[0];

#define mtVecAcc(devFunc, rInit, rExp, ...) \
    cudaGetLastError(); \
    size_t nb = divCeil(pX->nx, VEC_ACC_NTB * VEC_ACC_NPT); \
    devFunc<<<nb, VEC_ACC_NTB, 0, stream>>>(pX->p, ##__VA_ARGS__, pX->nx, buf); \
    syncCheck; \
    float r = (rInit); \
    for (size_t i = 0; i < nb; i++) { rExp; } \
    *pRes = r;

__global__ void devVecSum(float *x, size_t n, float *r) {
    devVecAcc(0, rb[itb] += rb[itb + i], rt += x[i]);
}

__export void mtVecSum(MtTensor *pX, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecSum, 0, r += buf[i]);
}

__global__ void devVecSquareSum(float *x, size_t n, float *r) {
    float e;
    devVecAcc(0, rb[itb] += rb[itb + i], e = x[i], rt += e * e);
}

__export void mtVecSquareSum(MtTensor *pX, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecSquareSum, 0, r += buf[i]);
}

__global__ void devVecMin(float *x, size_t n, float *r) {
    devVecAcc(NAN, rb[itb] = fminf(rb[itb], rb[itb + i]), rt = fminf(x[i], rt));
}

__export void mtVecMin(MtTensor *pX, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecMin, NAN, r = fminf(buf[i], r));
}

__global__ void devVecMax(float *x, size_t n, float *r) {
    devVecAcc(NAN, rb[itb] = fmaxf(rb[itb], rb[itb + i]), rt = fmaxf(x[i], rt));
}

__export void mtVecMax(MtTensor *pX, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecMax, NAN, r = fmaxf(buf[i], r));
}

__global__ void devVecDot(float *x, float *y, size_t n, float *r) {
    devVecAcc(0, rb[itb] += rb[itb + i], rt += x[i] * y[i]);
}

__export void mtVecDot(MtTensor *pX, MtTensor *pY, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecDot, 0, r += buf[i], pY->p);
}

__global__ void devVecSumSquareSum(float *x, float *y, size_t n, float *r) {
    float e;
    devVecAcc(0, rb[itb] += rb[itb + i], e = x[i] + y[i], rt += e * e);
}

__export void mtVecSumSquareSum(MtTensor *pX, MtTensor *pY, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecSumSquareSum, 0, r += buf[i], pY->p);
}

__global__ void devVecDiffSquareSum(float *x, float *y, size_t n, float *r) {
    float e;
    devVecAcc(0, rb[itb] += rb[itb + i], e = x[i] - y[i], rt += e * e);
}

__export void mtVecDiffSquareSum(MtTensor *pX, MtTensor *pY, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVecAcc(devVecDiffSquareSum, 0, r += buf[i], pY->p);
}
