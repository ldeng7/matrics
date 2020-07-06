#include "internal.h"

// ind

const size_t MAT_NTB_X = 32;
const size_t MAT_NTB_Y = 32;

#define devMatInd(...) \
    size_t i = size_t(blockIdx.x) * MAT_NTB_X + size_t(threadIdx.x); \
    size_t j = size_t(blockIdx.y) * MAT_NTB_Y + size_t(threadIdx.y); \
    if (i < n && j < m) { ##__VA_ARGS__; }

#define mtMatInd(devFunc, ...) \
    cudaGetLastError(); \
    size_t nbx = divCeil(pX->nx, MAT_NTB_X); \
    size_t nby = divCeil(pX->ny, MAT_NTB_Y); \
    devFunc<<<dim3(nbx, nby, 1), dim3(MAT_NTB_X, MAT_NTB_Y, 1), 0, stream>>>( \
        pX->p, pY->p, ##__VA_ARGS__, pX->ny, pX->nx); \
    syncCheck;

__global__ void devMatT(float *x, float *y, size_t m, size_t n) {
    devMatInd(y[m * i + j] = x[n * j + i]);
}

__export void mtMatT(MtTensor *pX, MtTensor *pY,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatT);
}

__global__ void devMatAddScalar(float *x, float *y, float a, size_t m, size_t n) {
    size_t k;
    devMatInd(k = n * j + i, y[k] = x[k] + a);
}

__export void mtMatAddScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatAddScalar, *pA);
}

__global__ void devMatMulScalar(float *x, float *y, float a, size_t m, size_t n) {
    size_t k;
    devMatInd(k = n * j + i, y[k] = a * x[k]);
}

__export void mtMatMulScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatMulScalar, *pA);
}

__global__ void devMatMulAddScalar(float *x, float *y, float a, float b, size_t m, size_t n) {
    size_t k;
    devMatInd(k = n * j + i, y[k] = a * x[k] + b);
}

__export void mtMatMulAddScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatMulAddScalar, *pA, *pB);
}

__global__ void devMatPowScalar(float *x, float *y, float a, size_t m, size_t n) {
    size_t k;
    devMatInd(k = n * j + i, y[k] = __powf(x[k], a));
}

__export void mtMatPowScalar(MtTensor *pX, MtTensor *pY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatPowScalar, *pA);
}

__global__ void devMatPowMulScalar(float *x, float *y, float a, float b, size_t m, size_t n) {
    size_t k;
    devMatInd(k = n * j + i, y[k] = __powf(x[k], a) * b);
}

__export void mtMatPowMulScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatPowMulScalar, *pA, *pB);
}

__global__ void devMatAddMat(float *x, float *y, float *z, size_t m, size_t n) {
    size_t k;
    devMatInd(k = n * j + i, z[k] = x[k] + y[k]);
}

__export void mtMatAddMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatAddMat, pZ->p);
}

__global__ void devMatSubMat(float *x, float *y, float *z, size_t m, size_t n) {
    size_t k;
    devMatInd(k = n * j + i, z[k] = x[k] - y[k]);
}

__export void mtMatSubMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    mtMatInd(devMatSubMat, pZ->p);
}

// matmul

__global__ void devMatMulMat(float *x, float *y, float *z, size_t m, size_t n, size_t l) {
    size_t i = size_t(blockIdx.x) * MAT_NTB_X + size_t(threadIdx.x);
    size_t j = size_t(blockIdx.y) * MAT_NTB_Y + size_t(threadIdx.y);
    if (i >= l || j >= m) return;
    float s = 0;
    size_t nj = n * j;
    for (size_t k = 0; k < n; k++) {
        s += x[nj + k] * y[l * k + i];
    }
    z[l * j + i] = s;
}

__export void mtMatMulMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    size_t nbx = divCeil(pY->nx, MAT_NTB_X);
    size_t nby = divCeil(pX->ny, MAT_NTB_Y);
    devMatMulMat<<<dim3(nbx, nby, 1), dim3(MAT_NTB_X, MAT_NTB_Y, 1), 0, stream>>>(
        pX->p, pY->p, pZ->p, pX->ny, pX->nx, pY->nx);
    syncCheck;
}

__global__ void devVecTMulMat(float *x, float *y, float *z, size_t n, size_t l) {
    size_t i = size_t(blockIdx.x) * MAT_NTB_X + size_t(threadIdx.x);
    if (i >= l) return;
    float s = 0;
    for (size_t k = 0; k < n; k++) {
        s += x[k] * y[l * k + i];
    }
    z[i] = s;
}

__export void mtVecTMulMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    devVecTMulMat<<<divCeil(pY->nx, MAT_NTB_X), MAT_NTB_X, 0, stream>>>(
        pX->p, pY->p, pZ->p, pX->nx, pY->nx);
    syncCheck;
}

__global__ void devMatMulVec(float *x, float *y, float *z, size_t m, size_t n) {
    size_t j = size_t(blockIdx.x) * MAT_NTB_Y + size_t(threadIdx.x);
    if (j >= m) return;
    float s = 0;
    size_t nj = n * j;
    for (size_t k = 0; k < n; k++) {
        s += x[nj + k] * y[k];
    }
    z[j] = s;
}

__export void mtMatMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    devMatMulVec<<<divCeil(pX->ny, MAT_NTB_Y), MAT_NTB_Y, 0, stream>>>(
        pX->p, pY->p, pZ->p, pX->ny, pX->nx);
    syncCheck;
}

__global__ void devMatMulMatAddVecTAct(float *x, float *y, float *z, float *w, size_t m, size_t n, size_t l,
        MtActivationType at, float aa) {
    size_t i = size_t(blockIdx.x) * MAT_NTB_X + size_t(threadIdx.x);
    size_t j = size_t(blockIdx.y) * MAT_NTB_Y + size_t(threadIdx.y);
    if (i >= l || j >= m) return;
    float s = 0;
    size_t nj = n * j;
    for (size_t k = 0; k < n; k++) {
        s += x[nj + k] * y[l * k + i];
    }
    s += z[i];
    activate(s, at, aa);
    w[l * j + i] = s;
}

__export void mtMatMulMatAddVecTAct(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtActivation *act,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    size_t nbx = divCeil(pY->nx, MAT_NTB_X);
    size_t nby = divCeil(pX->ny, MAT_NTB_Y);
    devMatMulMatAddVecTAct<<<dim3(nbx, nby, 1), dim3(MAT_NTB_X, MAT_NTB_Y, 1), 0, stream>>>(
        pX->p, pY->p, pZ->p, pW->p, pX->ny, pX->nx, pY->nx, act->typ, act->a);
    syncCheck;
}

__global__ void devVecTMulMatAddVecTAct(float *x, float *y, float *z, float *w, size_t n, size_t l,
        MtActivationType at, float aa) {
    size_t i = size_t(blockIdx.x) * MAT_NTB_X + size_t(threadIdx.x);
    if (i >= l) return;
    float s = 0;
    for (size_t k = 0; k < n; k++) {
        s += x[k] * y[l * k + i];
    }
    s += z[i];
    activate(s, at, aa);
    w[i] = s;
}

__export void mtVecTMulMatAddVecTAct(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtActivation *act,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    devVecTMulMatAddVecTAct<<<divCeil(pY->nx, MAT_NTB_X), MAT_NTB_X, 0, stream>>>(
        pX->p, pY->p, pZ->p, pW->p, pX->nx, pY->nx, act->typ, act->a);
    syncCheck;
}

const size_t MAT_ABN_NTB = 512;
const size_t MAT_ABN_NPT = 8;

__export void mtNewMatFlatBuffer(MtTensor *pTen, MtBuffer *pBuf, int *pCode) {
    *pBuf = nullptr;
    MtBuffer buf = mtNewBuffer(divCeil(pTen->nx, MAT_ABN_NTB * MAT_ABN_NPT) * pTen->ny, pCode);
    if (cudaSuccess != *pCode) return;
    *pBuf = buf;
}

__global__ void devMatFlatMulVec(float *x, float *y, size_t m, size_t n, float *r) {
    size_t iby = size_t(blockIdx.y);
    size_t itb = size_t(threadIdx.x);
    size_t i = size_t(blockIdx.x) * MAT_ABN_NTB * MAT_ABN_NPT + itb;
    float rb = 0;
    for (size_t j = 0; j < MAT_ABN_NPT && i < n; j++, i += MAT_ABN_NTB) {
        rb += x[n * iby + i] * y[i];
    }
    __shared__ float rbs[MAT_ABN_NTB];
    rbs[itb] = rb;
    __syncthreads();
    for (size_t i = MAT_ABN_NTB >> 1; i != 0; i >>= 1) {
        if (itb < i) rbs[itb] += rbs[itb + i];
        __syncthreads();
    }
    if (0 == itb) r[gridDim.x * iby + blockIdx.x] = rbs[0];
}

__export void mtMatFlatMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtBuffer buf,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    size_t nbx = divCeil(pX->nx, MAT_ABN_NTB * MAT_ABN_NPT);
    devMatFlatMulVec<<<dim3(nbx, pX->ny, 1), MAT_ABN_NTB, 0, stream>>>(
        pX->p, pY->p, pX->ny, pX->nx, buf);
    syncCheck;
    float *z = pZ->p;
    for (size_t i = 0; i < pZ->nx; i++) {
        float r = 0;
        float *pr = buf + nbx * i;
        float *pre = pr + nbx;
        for (; pr != pre; pr++) {
            r += *pr;
        }
        z[i] = r;
    }
}
