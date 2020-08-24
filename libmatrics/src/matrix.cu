#include "internal.h"

// ind

const uint32 NTB_X = 32;
const uint32 NTB_Y = 32;

#define devMatInd(...) \
    uint32 i = blockIdx.x * NTB_X + threadIdx.x; \
    uint32 j = blockIdx.y * NTB_Y + threadIdx.y; \
    if (i < n && j < m) { __VA_ARGS__; }

#define mtMatInd(devFunc, ...) \
    uint32 nbx = divCeil(pX->nx, NTB_X); \
    uint32 nby = divCeil(pX->ny, NTB_Y); \
    cudaGetLastError(); \
    devFunc<<<dim3(nbx, nby, 1), dim3(NTB_X, NTB_Y, 1), 0, stream>>>( \
        pX->p, pX->ny, pX->nx, __VA_ARGS__); \
    syncCheck(stream, pCode);

__global__ void devMatT(float *x, uint32 m, uint32 n, float *y) {
    devMatInd(y[m * i + j] = x[n * j + i]);
}

__export void mtMatT(MtTensor *pX, MtTensor *pY,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatT, pY->p);
}

__global__ void devMatAddScalar(float *x, uint32 m, uint32 n, float *y, float a) {
    uint32 k;
    devMatInd(k = n * j + i, y[k] = x[k] + a);
}

__export void mtMatAddScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatAddScalar, pY->p, *pA);
}

__global__ void devMatMulScalar(float *x, uint32 m, uint32 n, float *y, float a) {
    uint32 k;
    devMatInd(k = n * j + i, y[k] = a * x[k]);
}

__export void mtMatMulScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatMulScalar, pY->p, *pA);
}

__global__ void devMatMulAddScalar(float *x, uint32 m, uint32 n, float *y, float a, float b) {
    uint32 k;
    devMatInd(k = n * j + i, y[k] = a * x[k] + b);
}

__export void mtMatMulAddScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatMulAddScalar, pY->p, *pA, *pB);
}

__global__ void devMatPowScalar(float *x, uint32 m, uint32 n, float *y, float a) {
    uint32 k;
    devMatInd(k = n * j + i, y[k] = __powf(x[k], a));
}

__export void mtMatPowScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatPowScalar, pY->p, *pA);
}

__global__ void devMatPowMulScalar(float *x, uint32 m, uint32 n, float *y, float a, float b) {
    uint32 k;
    devMatInd(k = n * j + i, y[k] = __powf(x[k], a) * b);
}

__export void mtMatPowMulScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatPowMulScalar, pY->p, *pA, *pB);
}

__global__ void devMatAddMat(float *x, uint32 m, uint32 n, float *y, float *z) {
    uint32 k;
    devMatInd(k = n * j + i, z[k] = x[k] + y[k]);
}

__export void mtMatAddMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatAddMat, pY->p, pZ->p);
}

__global__ void devMatSubMat(float *x, uint32 m, uint32 n, float *y, float *z) {
    uint32 k;
    devMatInd(k = n * j + i, z[k] = x[k] - y[k]);
}

__export void mtMatSubMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatSubMat, pY->p, pZ->p);
}

// matmul

__global__ void devMatMulMat(float *x, float *y, float *z, uint32 m, uint32 n, uint32 l) {
    uint32 i = blockIdx.x * NTB_X + threadIdx.x;
    uint32 j = blockIdx.y * NTB_Y + threadIdx.y;
    if (i >= l || j >= m) return;
    float s = 0;
    uint32 nj = n * j;
    for (uint32 k = 0; k < n; k++) {
        s += x[nj + k] * y[l * k + i];
    }
    z[l * j + i] = s;
}

__export void mtMatMulMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    uint32 nbx = divCeil(pY->nx, NTB_X);
    uint32 nby = divCeil(pX->ny, NTB_Y);
    cudaGetLastError();
    devMatMulMat<<<dim3(nbx, nby, 1), dim3(NTB_X, NTB_Y, 1), 0, stream>>>(
        pX->p, pY->p, pZ->p, pX->ny, pX->nx, pY->nx);
    syncCheck(stream, pCode);
}

__global__ void devVecTMulMat(float *x, float *y, float *z, uint32 n, uint32 l) {
    uint32 i = blockIdx.x * NTB_X + threadIdx.x;
    if (i >= l) return;
    float s = 0;
    for (uint32 k = 0; k < n; k++) {
        s += x[k] * y[l * k + i];
    }
    z[i] = s;
}

__export void mtVecTMulMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    devVecTMulMat<<<divCeil(pY->nx, NTB_X), NTB_X, 0, stream>>>(
        pX->p, pY->p, pZ->p, pX->nx, pY->nx);
    syncCheck(stream, pCode);
}

__global__ void devMatMulVec(float *x, float *y, float *z, uint32 m, uint32 n) {
    uint32 j = blockIdx.x * NTB_Y + threadIdx.x;
    if (j >= m) return;
    float s = 0;
    uint32 nj = n * j;
    for (uint32 k = 0; k < n; k++) {
        s += x[nj + k] * y[k];
    }
    z[j] = s;
}

__export void mtMatMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    devMatMulVec<<<divCeil(pX->ny, NTB_Y), NTB_Y, 0, stream>>>(
        pX->p, pY->p, pZ->p, pX->ny, pX->nx);
    syncCheck(stream, pCode);
}
