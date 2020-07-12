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

void newBuffer(uint32 sz, buffer *pBuf, int *pCode) {
    *pBuf = nullptr;
    buffer buf;
    if (cudaSuccess != (*pCode = cudaMallocManaged(&buf, sz))) return;
    *pBuf = buf;
}

__export void mtNewBuffer(uint32 sz, buffer *pBuf, int *pCode) {
    newBuffer(sz, pBuf, pCode);
}

__export void mtBufferDestroy(buffer buf) {
    cudaFree(buf);
}
