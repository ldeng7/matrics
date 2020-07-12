#pragma once

#include "../include/common.h"

typedef void *buffer;

#ifdef _WIN64
    #define __export extern "C" __declspec(dllexport)
#elif __linux__
    #define __export extern "C"
#endif

#define divCeil(a, b) (((a) + (b) - 1) / (b))

#define syncCheck(stream, pCode) \
    *pCode = cudaSuccess; \
    cudaError_t code = cudaGetLastError(); \
    cudaError_t code1 = cudaStreamSynchronize(stream); \
    if (cudaSuccess == code) code = code1; \
    if (cudaSuccess != code) { \
        *pCode = code; \
        return; \
    }

extern void newBuffer(uint32 sz, buffer *pBuf, int *pCode);
