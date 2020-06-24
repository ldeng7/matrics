#include "internal.h"

__export void mtNewMatrix(size_t w, size_t h, MtMatrix **ppMat, float **ppBuf, int *pCode) {
    *ppMat = nullptr;
    MtMatrix *pMat = (MtMatrix *)(malloc(sizeof(MtMatrix)));
    if (nullptr == pMat) {
        *pCode = cudaErrorApiFailureBase;
        return;
    }
    pMat->w = w;
    pMat->h = h;
    size_t sz = sizeof(float) * w * h;
    if (cudaSuccess != (*pCode = cudaMallocManaged(&(pMat->p), sz))) {
        free(pMat);
        return;
    }
    *ppMat = pMat;
    *ppBuf = pMat->p;
}

__export void mtMatrixDestroy(MtMatrix *pMat) {
    if (nullptr != pMat->p) cudaFree(pMat->p);
    free(pMat);
}

const size_t MAT_NTB_X = 32;
const size_t MAT_NTB_Y = 32;

#define devMatrixInd(...) \
    size_t i = size_t(blockIdx.x) * MAT_NTB_X + size_t(threadIdx.x); \
    size_t j = size_t(blockIdx.y) * MAT_NTB_Y + size_t(threadIdx.y); \
    if (i < n && j < m) { ##__VA_ARGS__; }

#define mtMatrixInd(devFunc, ...) \
    cudaGetLastError(); \
    size_t nbx = divCeil(pMatX->w, MAT_NTB_X); \
    size_t nby = divCeil(pMatX->h, MAT_NTB_Y); \
    devFunc<<<dim3(nbx, nby, 1), dim3(MAT_NTB_X, MAT_NTB_Y, 1), 0, stream>>>( \
        pMatX->p, pMatY->p, ##__VA_ARGS__, pMatX->h, pMatX->w); \
    syncCheck;

__global__ void devMatrixT(float *x, float *y, size_t m, size_t n) {
    devMatrixInd(y[m * i + j] = x[n * j + i]);
}

__export void mtMatrixT(MtMatrix *pMatX, MtMatrix *pMatY,
        cudaStream_t stream, int *pCode) {
    mtMatrixInd(devMatrixT);
}

__global__ void devMatrixAddScalar(float *x, float *y, float a, size_t m, size_t n) {
    size_t k;
    devMatrixInd(k = n * j + i, y[k] = x[k] + a);
}

__export void mtMatrixAddScalar(MtMatrix *pMatX, MtMatrix *pMatY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtMatrixInd(devMatrixAddScalar, *pA);
}

__global__ void devMatrixMulScalar(float *x, float *y, float a, size_t m, size_t n) {
    size_t k;
    devMatrixInd(k = n * j + i, y[k] = a * x[k]);
}

__export void mtMatrixMulScalar(MtMatrix *pMatX, MtMatrix *pMatY, float *pA,
        cudaStream_t stream, int *pCode) {
    mtMatrixInd(devMatrixMulScalar, *pA);
}

__global__ void devMatrixMulAddScalar(float *x, float *y, float a, float b, size_t m, size_t n) {
    size_t k;
    devMatrixInd(k = n * j + i, y[k] = a * x[k] + b);
}

__export void mtMatrixMulAddScalar(MtMatrix *pMatX, MtMatrix *pMatY, float *pA, float *pB,
        cudaStream_t stream, int *pCode) {
    mtMatrixInd(devMatrixMulAddScalar, *pA, *pB);
}

__global__ void devMatrixAddMatrix(float *x, float *y, float *z, size_t m, size_t n) {
    size_t k;
    devMatrixInd(k = n * j + i, z[k] = x[k] + y[k]);
}

__export void mtMatrixAddMatrix(MtMatrix *pMatX, MtMatrix *pMatY, MtMatrix *pMatZ,
        cudaStream_t stream, int *pCode) {
    mtMatrixInd(devMatrixAddMatrix, pMatZ->p);
}

__global__ void devMatrixSubMatrix(float *x, float *y, float *z, size_t m, size_t n) {
    size_t k;
    devMatrixInd(k = n * j + i, z[k] = x[k] - y[k]);
}

__export void mtMatrixSubMatrix(MtMatrix *pMatX, MtMatrix *pMatY, MtMatrix *pMatZ,
        cudaStream_t stream, int *pCode) {
    mtMatrixInd(devMatrixSubMatrix, pMatZ->p);
}

__global__ void devMatrixMulMatrix(float *x, float *y, float *z, size_t m, size_t n, size_t l) {
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

__export void mtMatrixMulMatrix(MtMatrix *pMatX, MtMatrix *pMatY, MtMatrix *pMatZ,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    size_t nbx = divCeil(pMatZ->w, MAT_NTB_X);
    size_t nby = divCeil(pMatZ->h, MAT_NTB_Y);
    devMatrixMulMatrix<<<dim3(nbx, nby, 1), dim3(MAT_NTB_X, MAT_NTB_Y, 1), 0, stream>>>(
        pMatX->p, pMatY->p, pMatZ->p, pMatX->h, pMatX->w, pMatY->w);
    syncCheck;
}

__global__ void devMatrixMulVector(float *x, float *y, float *z, size_t m, size_t n) {
    size_t j = size_t(blockIdx.x) * MAT_NTB_Y + size_t(threadIdx.x);
    if (j >= m) return;
    float s = 0;
    size_t nj = n * j;
    for (size_t k = 0; k < n; k++) {
        s += x[nj + k] * y[k];
    }
    z[j] = s;
}

__export void mtMatrixMulVector(MtMatrix *pMatX, MtVector *pVecY, MtVector *pVecZ,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    devMatrixMulVector<<<divCeil(pVecZ->n, MAT_NTB_Y), MAT_NTB_Y, 0, stream>>>(
        pMatX->p, pVecY->p, pVecZ->p, pMatX->h, pMatX->w);
    syncCheck;
}

const size_t MAT_W_NTB = 512;
const size_t MAT_W_NPT = 8;

__export void mtNewMatrixWideBuffer(MtMatrix *pMat, MtBuffer *pBuf, int *pCode) {
    *pBuf = nullptr;
	MtBuffer buf = mtNewBuffer(divCeil(pMat->w, MAT_W_NTB * MAT_W_NPT) * pMat->h, pCode);
    if (cudaSuccess != *pCode) return;
    *pBuf = buf;
}

__global__ void devMatrixWideMulVector(float *x, float *y, size_t m, size_t n, float *r) {
    size_t iby = size_t(blockIdx.y);
    size_t itb = size_t(threadIdx.x);
    size_t i = size_t(blockIdx.x) * MAT_W_NTB * MAT_W_NPT + itb;
    float rb = 0;
    for (size_t j = 0; j < MAT_W_NPT && i < n; j++, i += MAT_W_NTB) {
        rb += x[n * iby + i] * y[i];
    }
    __shared__ float rbs[MAT_W_NTB];
    rbs[itb] = rb;
    __syncthreads();
    for (size_t i = MAT_W_NTB >> 1; i != 0; i >>= 1) {
        if (itb < i) rbs[itb] += rbs[itb + i];
        __syncthreads();
    }
    if (0 == itb) r[gridDim.x * iby + blockIdx.x] = rbs[0];
}

__export void mtMatrixWideMulVector(MtMatrix *pMatX, MtVector *pVecY, MtVector *pVecZ, MtBuffer buf,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    size_t nbx = divCeil(pMatX->w, MAT_W_NTB * MAT_W_NPT);
    devMatrixWideMulVector<<<dim3(nbx, pMatX->h, 1), MAT_W_NTB, 0, stream>>>(
        pMatX->p, pVecY->p, pMatX->h, pMatX->w, buf);
    syncCheck;
    float *z = pVecZ->p;
    for (size_t i = 0; i < pVecZ->n; i++) {
    	float r = 0;
		float *pr = buf + nbx * i;
		float *pre = pr + nbx;
		for (; pr != pre; pr++) {
			r += *pr;
		}
		z[i] = r;
    }
}
