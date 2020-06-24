#include "internal.h"

__export void mtNewVector(size_t n, MtVector **ppVec, float **ppBuf, int *pCode) {
    *ppVec = nullptr;
    MtVector *pVec = (MtVector *)(malloc(sizeof(MtVector)));
    if (nullptr == pVec) {
        *pCode = cudaErrorApiFailureBase;
        return;
    }
    pVec->n = n;
    size_t sz = sizeof(float) * n;
    if (cudaSuccess != (*pCode = cudaMallocManaged(&(pVec->p), sz))) {
        free(pVec);
        return;
    }
    *ppVec = pVec;
    *ppBuf = pVec->p;
}

__export void mtVectorDestroy(MtVector *pVec) {
    if (nullptr != pVec->p) cudaFree(pVec->p);
    free(pVec);
}

// ind

const size_t VEC_IND_NTB = 512;

#define devVectorInd(...) \
    size_t i = size_t(blockIdx.x) * VEC_IND_NTB + size_t(threadIdx.x); \
    if (i < n) { ##__VA_ARGS__; }

#define mtVectorInd(devFunc, ...) \
    cudaGetLastError(); \
    devFunc<<<divCeil(pVecX->n, VEC_IND_NTB), VEC_IND_NTB, 0, stream>>>( \
        pVecX->p, pVecY->p, ##__VA_ARGS__, pVecX->n); \
    syncCheck;

__global__ void devVectorAddScalar(float *x, float *y, float a, size_t n) {
    devVectorInd(y[i] = x[i] + a);
}

__export void mtVectorAddScalar(MtVector *pVecX, MtVector *pVecY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVectorInd(devVectorAddScalar, *pA);
}

__global__ void devVectorMulScalar(float *x, float *y, float a, size_t n) {
    devVectorInd(y[i] = a * x[i]);
}

__export void mtVectorMulScalar(MtVector *pVecX, MtVector *pVecY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVectorInd(devVectorMulScalar, *pA);
}

__global__ void devVectorMulAddScalar(float *x, float *y, float a, float b, size_t n) {
    devVectorInd(y[i] = a * x[i] + b);
}

__export void mtVectorMulAddScalar(MtVector *pVecX, MtVector *pVecY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtVectorInd(devVectorMulAddScalar, *pA, *pB);
}

__global__ void devVectorPowScalar(float *x, float *y, float a, size_t n) {
    devVectorInd(y[i] = __powf(x[i], a));
}

__export void mtVectorPowScalar(MtVector *pVecX, MtVector *pVecY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtVectorInd(devVectorPowScalar, *pA);
}

__global__ void devVectorPowMulScalar(float *x, float *y, float a, float b, size_t n) {
    devVectorInd(y[i] = __powf(x[i], a) * b);
}

__export void mtVectorPowMulScalar(MtVector *pVecX, MtVector *pVecY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtVectorInd(devVectorPowMulScalar, *pA, *pB);
}

__global__ void devVectorAddVector(float *x, float *y, float *z, size_t n) {
    devVectorInd(z[i] = x[i] + y[i]);
}

__export void mtVectorAddVector(MtVector *pVecX, MtVector *pVecY, MtVector *pVecZ,
        cudaStream_t stream, int *pCode) {
    mtVectorInd(devVectorAddVector, pVecZ->p);
}

__global__ void devVectorSubVector(float *x, float *y, float *z, size_t n) {
    devVectorInd(z[i] = x[i] - y[i]);
}

__export void mtVectorSubVector(MtVector *pVecX, MtVector *pVecY, MtVector *pVecZ,
        cudaStream_t stream, int *pCode) {
    mtVectorInd(devVectorSubVector, pVecZ->p);
}

__global__ void devVectorPatchMulVector(float *x, float *y, float *z, size_t n) {
    devVectorInd(z[i] = x[i] * y[i]);
}

__export void mtVectorPatchMulVector(MtVector *pVecX, MtVector *pVecY, MtVector *pVecZ,
        cudaStream_t stream, int *pCode) {
    mtVectorInd(devVectorPatchMulVector, pVecZ->p);
}

__global__ void devVectorTMulMatrix(float *x, float *y, float *z, size_t n, size_t l) {
    size_t i = size_t(blockIdx.x) * blockDim.x + size_t(threadIdx.x);
    if (i >= l) return;
    float s = 0;
    for (size_t k = 0; k < n; k++) {
        s += x[k] * y[l * k + i];
    }
    z[i] = s;
}

__export void mtVectorTMulMatrix(MtVector *pVecX, MtMatrix *pMatY, MtVector *pVecZ,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    const size_t ntb = 32;
    devVectorTMulMatrix<<<divCeil(pVecZ->n, ntb), ntb, 0, stream>>>(
        pVecX->p, pMatY->p, pVecZ->p, pVecX->n, pVecZ->n);
    syncCheck;
}

// acc

const size_t VEC_ACC_NTB = 512;
const size_t VEC_ACC_NPT = 8;

__export void mtNewVectorAccBuffer(MtVector *pVec, MtBuffer *pBuf, int *pCode) {
    *pBuf = nullptr;
	MtBuffer buf = mtNewBuffer(divCeil(pVec->n, VEC_ACC_NTB * VEC_ACC_NPT), pCode);
    if (cudaSuccess != *pCode) return;
    *pBuf = buf;
}

#define devVectorAcc(rbInit, rbsExp, ...) \
    size_t itb = size_t(threadIdx.x); \
    size_t i = size_t(blockIdx.x) * VEC_ACC_NTB * VEC_ACC_NPT + itb; \
    float rb = (rbInit); \
    for (size_t j = 0; j < VEC_ACC_NPT && i < n; j++, i += VEC_ACC_NTB) { ##__VA_ARGS__; } \
    __shared__ float rbs[VEC_ACC_NTB]; \
    rbs[itb] = rb; \
    __syncthreads(); \
    for (size_t i = VEC_ACC_NTB >> 1; i != 0; i >>= 1) { \
        if (itb < i) { rbsExp; } \
        __syncthreads(); \
    } \
    if (0 == itb) r[blockIdx.x] = rbs[0];

#define mtVectorAcc(devFunc, rInit, rExp, ...) \
    cudaGetLastError(); \
    size_t nb = divCeil(pVecX->n, VEC_ACC_NTB * VEC_ACC_NPT); \
    devFunc<<<nb, VEC_ACC_NTB, 0, stream>>>(pVecX->p, ##__VA_ARGS__, pVecX->n, buf); \
    syncCheck; \
    float r = (rInit); \
    for (size_t i = 0; i < nb; i++) { rExp; } \
    *pRes = r;

__global__ void devVectorSum(float *x, size_t n, float *r) {
    devVectorAcc(0, rbs[itb] += rbs[itb + i], rb += x[i]);
}

__export void mtVectorSum(MtVector *pVecX, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVectorAcc(devVectorSum, 0, r += buf[i]);
}

__global__ void devVectorSquareSum(float *x, size_t n, float *r) {
    float e;
    devVectorAcc(0, rbs[itb] += rbs[itb + i], e = x[i], rb += e * e);
}

__export void mtVectorSquareSum(MtVector *pVecX, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVectorAcc(devVectorSquareSum, 0, r += buf[i]);
}

__global__ void devVectorMin(float *x, size_t n, float *r) {
    devVectorAcc(NAN, rbs[itb] = fminf(rbs[itb], rbs[itb + i]), rb = fminf(x[i], rb));
}

__export void mtVectorMin(MtVector *pVecX, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVectorAcc(devVectorMin, NAN, r = fminf(buf[i], r));
}

__global__ void devVectorMax(float *x, size_t n, float *r) {
    devVectorAcc(NAN, rbs[itb] = fmaxf(rbs[itb], rbs[itb + i]), rb = fmaxf(x[i], rb));
}

__export void mtVectorMax(MtVector *pVecX, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVectorAcc(devVectorMax, NAN, r = fmaxf(buf[i], r));
}

__global__ void devVectorDot(float *x, float *y, size_t n, float *r) {
    devVectorAcc(0, rbs[itb] += rbs[itb + i], rb += x[i] * y[i]);
}

__export void mtVectorDot(MtVector *pVecX, MtVector *pVecY, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVectorAcc(devVectorDot, 0, r += buf[i], pVecY->p);
}

__global__ void devVectorSumSquareSum(float *x, float *y, size_t n, float *r) {
    float e;
    devVectorAcc(0, rbs[itb] += rbs[itb + i], e = x[i] + y[i], rb += e * e);
}

__export void mtVectorSumSquareSum(MtVector *pVecX, MtVector *pVecY, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVectorAcc(devVectorSumSquareSum, 0, r += buf[i], pVecY->p);
}

__global__ void devVectorDiffSquareSum(float *x, float *y, size_t n, float *r) {
    float e;
    devVectorAcc(0, rbs[itb] += rbs[itb + i], e = x[i] - y[i], rb += e * e);
}

__export void mtVectorDiffSquareSum(MtVector *pVecX, MtVector *pVecY, MtBuffer buf,
        cudaStream_t stream, float *pRes, int *pCode) {
    mtVectorAcc(devVectorDiffSquareSum, 0, r += buf[i], pVecY->p);
}
