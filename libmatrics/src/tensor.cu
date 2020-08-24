#include "internal.h"

__export void mtNewTensor(uint32 nx, uint32 ny, uint32 nz, uint32 nw,
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
    uint32 n = nx * ny * nz * nw;
    if (cudaSuccess != (*pCode = cudaMallocManaged(&(pTen->p), sizeof(float) * n))) {
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
