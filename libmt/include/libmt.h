#pragma once

#include "common.h"

#define mt_import extern __declspec(dllimport)

mt_import void mtNewStream(size_t *pStream, int *pCode);
mt_import void mtStreamDestroy(size_t stream);
mt_import void mtNewTensor(size_t nx, size_t ny, size_t nz, size_t nw, MtTensor **ppTen, float **ppBuf, int *pCode);
mt_import void mtTensorDestroy(MtTensor *pTen);
mt_import void mtBufferDestroy(MtBuffer buf);

// vector

mt_import void mtVecAddScalar(MtTensor *pX, MtTensor *pY, float *pA, size_t stream, int *pCode);
mt_import void mtVecMulScalar(MtTensor *pX, MtTensor *pY, float *pA, size_t stream, int *pCode);
mt_import void mtVecMulAddScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB, size_t stream, int *pCode);
mt_import void mtVecPowScalar(MtTensor *pX, MtTensor *pY, float *pA, size_t stream, int *pCode);
mt_import void mtVecPowMulScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB, size_t stream, int *pCode);
mt_import void mtVecAddVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);
mt_import void mtVecSubVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);
mt_import void mtVecPatchMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);

mt_import void mtNewVecAccBuffer(MtTensor *pTen, MtBuffer *pBuf, int *pCode);
mt_import void mtVecSum(MtTensor *pX, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVecSquareSum(MtTensor *pX, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVecMin(MtTensor *pX, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVecMax(MtTensor *pX, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVecDot(MtTensor *pX, MtTensor *pY, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVecSumSquareSum(MtTensor *pX, MtTensor *pY, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVecDiffSquareSum(MtTensor *pX, MtTensor *pY, MtBuffer buf, size_t stream, float *pRes, int *pCode);

// matrix

mt_import void mtMatT(MtTensor *pX, MtTensor *pY, cudaStream_t stream, int *pCode);
mt_import void mtMatAddScalar(MtTensor *pX, MtTensor *pY, float *pA, size_t stream, int *pCode);
mt_import void mtMatMulScalar(MtTensor *pX, MtTensor *pY, float *pA, size_t stream, int *pCode);
mt_import void mtMatMulAddScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB, size_t stream, int *pCode);
mt_import void mtMatPowScalar(MtTensor *pX, MtTensor *pY, float *pA, size_t stream, int *pCode);
mt_import void mtMatPowMulScalar(MtTensor *pX, MtTensor *pY, float *pA, float *pB, size_t stream, int *pCode);
mt_import void mtMatAddMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);
mt_import void mtMatSubMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);

mt_import void mtMatMulMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);
mt_import void mtVecTMulMat(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);
mt_import void mtMatMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);
mt_import void mtMatMulMatAddVecTAct(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, act *MtActivation, size_t stream, int *pCode);
mt_import void mtVecTMulMatAddVecTAct(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, act *MtActivation, size_t stream, int *pCode);
mt_import void mtNewMatFlatBuffer(MtTensor *pTen, MtBuffer *pBuf, int *pCode);
mt_import void mtMatFlatMulVec(MtTensor *pX, MtTensor *pY, MtTensor *pZ, size_t stream, int *pCode);

// tesseract

// input/output: {x: depth, y: width, z: height}
// core: {x: depth_in, y: width, z: height, w: depth_out}
mt_import void mtCubConv2d(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, conf *MtConv2dConf, size_t stream, int *pCode);
// input/output: {x: depth, y: width, z: height, w: batch}
mt_import void mtTesConv2d(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, conf *MtConv2dConf, size_t stream, int *pCode);
mt_import void mtCubCnnPool(MtTensor *pX, MtTensor *pY, MtCnnPoolConf *conf, size_t stream, int *pCode);
mt_import void mtTesCnnPool(MtTensor *pX, MtTensor *pY, MtCnnPoolConf *conf, size_t stream, int *pCode);
