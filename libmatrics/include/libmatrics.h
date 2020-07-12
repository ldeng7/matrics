#pragma once

#include "common.h"

#ifdef _WIN64
    #define mt_import extern __declspec(dllimport)
#elif __linux__
    #define mt_import extern
#endif

mt_import void mtNewStream(MtStream *pStream, int *pCode);
mt_import void mtStreamDestroy(MtStream stream);
mt_import void mtNewBuffer(uint32 sz, MtBuffer *pBuf, int *pCode);
mt_import void mtBufferDestroy(MtBuffer buf);
mt_import void mtNewTensor(uint32 nx, uint32 ny, uint32 nz, uint32 nw, MtTensor **ppTen, float **ppBuf, int *pCode);
mt_import void mtTensorDestroy(MtTensor *pTen);

// vector

mt_import void mtVecAddScalar(MtTensor *pX, MtTensor *pY, float *pA, MtStream stream, int *pCode);
mt_import void mtVecMulScalar(MtTensor *pX, MtTensor *pY, float *pA, MtStream stream, int *pCode);
mt_import void mtVecMulAddScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB, MtStream stream, int *pCode);
mt_import void mtVecPowScalar(MtTensor *pX, MtTensor *pY, float *pA, MtStream stream, int *pCode);
mt_import void mtVecPowMulScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB, MtStream stream, int *pCode);
mt_import void mtVecAddVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtStream stream, int *pCode);
mt_import void mtVecSubVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtStream stream, int *pCode);
mt_import void mtVecPatchMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtStream stream, int *pCode);

mt_import void mtNewVecAccBuffer(MtTensor *pTen, MtBuffer *pBuf, int *pCode);
mt_import void mtVecSum(MtTensor *pX, MtBuffer buf, MtStream stream, float *pRes, int *pCode);
mt_import void mtVecSquareSum(MtTensor *pX, MtBuffer buf, MtStream stream, float *pRes, int *pCode);
mt_import void mtVecMin(MtTensor *pX, MtBuffer buf, MtStream stream, float *pRes, int *pCode);
mt_import void mtVecMax(MtTensor *pX, MtBuffer buf, MtStream stream, float *pRes, int *pCode);
mt_import void mtVecDot(MtTensor *pX, MtTensor *pY, MtBuffer buf, MtStream stream, float *pRes, int *pCode);
mt_import void mtVecSumSquareSum(MtTensor *pX, MtTensor *pY, MtBuffer buf, MtStream stream, float *pRes, int *pCode);
mt_import void mtVecDiffSquareSum(MtTensor *pX, MtTensor *pY, MtBuffer buf, MtStream stream, float *pRes, int *pCode);

// matrix

mt_import void mtMatT(MtTensor *pX, MtTensor *pY, MtStream stream, int *pCode);
mt_import void mtMatAddScalar(MtTensor *pX, MtTensor *pY, float *pA, MtStream stream, int *pCode);
mt_import void mtMatMulScalar(MtTensor *pX, MtTensor *pY, float *pA, MtStream stream, int *pCode);
mt_import void mtMatMulAddScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB, MtStream stream, int *pCode);
mt_import void mtMatPowScalar(MtTensor *pX, MtTensor *pY, float *pA, MtStream stream, int *pCode);
mt_import void mtMatPowMulScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB, MtStream stream, int *pCode);
mt_import void mtMatAddMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtStream stream, int *pCode);
mt_import void mtMatSubMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtStream stream, int *pCode);
mt_import void mtMatMulMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtStream stream, int *pCode);
mt_import void mtVecTMulMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtStream stream, int *pCode);
mt_import void mtMatMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtStream stream, int *pCode);

// neural

mt_import void mtMatMulMatAddVecTAct(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtNnActivation *act, MtStream stream, int *pCode);
mt_import void mtVecTMulMatAddVecTAct(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtNnActivation *act, MtStream stream, int *pCode);
// input/output: {x: width, y: height, z: depth, w: batch}
// core: {x: width, y: height, z: depth_in, w: depth_out}
mt_import void mtCubBatchConv2d(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtConv2dConf *conf, MtStream stream, int *pCode);
mt_import void mtCubBatchPool2d(MtTensor *pX, MtTensor *pY, MtPool2dConf *conf, MtStream stream, int *pCode);
