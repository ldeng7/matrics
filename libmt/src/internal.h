#pragma once

#include "..\include\common.h"

#define __export extern "C" __declspec(dllexport)

#define divCeil(a, b) (((a) + (b) - 1) / (b))

#define syncCheck \
    *pCode = cudaSuccess; \
    cudaError_t code = cudaGetLastError(); \
    cudaError_t code1 = cudaStreamSynchronize(stream); \
    if (cudaSuccess == code) code = code1; \
    if (cudaSuccess != code) { \
    	*pCode = code; \
		return; \
    }

extern MtBuffer mtNewBuffer(size_t n, int *pCode);
