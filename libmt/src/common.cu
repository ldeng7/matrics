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

MtBuffer mtNewBuffer(size_t n, int *pCode) {
    MtBuffer buf;
    size_t sz = sizeof(float) * n;
    *pCode = cudaMallocManaged(&buf, sz);
    return buf;
}

__export void mtBufferDestroy(MtBuffer buf) {
    cudaFree(buf);
}
