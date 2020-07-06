#include "internal.h"

__export void mtNewStream(cudaStream_t *pStream, int *pCode) {
    *pStream = nullptr;
    cudaStream_t stream;
    if (cudaSuccess != (*pCode = cudaStreamCreate(&stream))) return;
    *pStream = stream;
}

__export void mtStreamDestroy(cudaStream_t stream) {
    cudaStreamDestroy(stream);
}

__export void mtNewTensor(size_t nx, size_t ny, size_t nz, size_t nw,
        MtTensor **ppTen, float **ppBuf, int *pCode) {
    *ppTen = nullptr;
    MtTensor *pTen = (MtTensor *)(malloc(sizeof(MtTensor)));
    if (nullptr == pTen) {
        *pCode = cudaErrorApiFailureBase;
        return;
    }
    pTen->nx = nx;
    pTen->ny = ny;
    pTen->nz = nz;
    pTen->nw = nw;
    size_t sz = sizeof(float) * nx * ny * nz * nw;
    if (cudaSuccess != (*pCode = cudaMallocManaged(&(pTen->p), sz))) {
        free(pTen);
        return;
    }
    *ppTen = pTen;
    *ppBuf = pTen->p;
}

__export void mtTensorDestroy(MtTensor *pTen) {
    if (nullptr != pTen->p) cudaFree(pTen->p);
    free(pTen);
}

MtBuffer mtNewBuffer(size_t n, int *pCode) {
    MtBuffer buf;
    size_t sz = sizeof(float) * n;
    *pCode = cudaMallocManaged(&buf, sz);
    return buf;
}

__export void mtBufferDestroy(MtBuffer buf) {
    cudaFree(buf);
}
