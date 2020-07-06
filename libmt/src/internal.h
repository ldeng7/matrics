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

#define activate(e, typ, a) \
    switch (typ) { \
    case MtActivationTypeReLU: e = (e < 0) ? 0 : e; break; \
    case MtActivationTypeLReLU: e = (e < 0) ? a * e : e; break; \
    case MtActivationTypeELU: e = (e < 0) ? a * (__expf(e) - 1) : 0; break; \
    case MtActivationTypeSwish: e = (e < 0) ? e / (1 + __expf(-a * e)) : 0; break; \
    }

extern MtBuffer mtNewBuffer(size_t n, int *pCode);
