#pragma once

#include "common.h"

#define mt_import extern __declspec(dllimport)

mt_import void mtNewStream(size_t *pStream, int *pCode);
mt_import void mtStreamDestroy(size_t stream);
mt_import void mtBufferDestroy(MtBuffer buf);

mt_import void mtNewVector(size_t n, MtVector **ppVec, float **ppBuf, int *pCode);
mt_import void mtVectorDestroy(MtVector *pVec);

mt_import void mtVectorAddScalar(MtVector *pVecX, MtVector *pVecY, float *pA, size_t stream, int *pCode);
mt_import void mtVectorMulScalar(MtVector *pVecX, MtVector *pVecY, float *pA, size_t stream, int *pCode);
mt_import void mtVectorMulAddScalar(MtVector *pVecX, MtVector *pVecY, float *pA, float *pB, size_t stream, int *pCode);
mt_import void mtVectorPowScalar(MtVector *pVecX, MtVector *pVecY, float *pA, size_t stream, int *pCode);
mt_import void mtVectorPowMulScalar(MtVector *pVecX, MtVector *pVecY, float *pA, float *pB, size_t stream, int *pCode);
mt_import void mtVectorAddVector(MtVector *pVecX, MtVector *pVecY, MtVector *pVecZ, size_t stream, int *pCode);
mt_import void mtVectorSubVector(MtVector *pVecX, MtVector *pVecY, MtVector *pVecZ, size_t stream, int *pCode);
mt_import void mtVectorPatchMulVector(MtVector *pVecX, MtVector *pVecY, MtVector *pVecZ, size_t stream, int *pCode);
mt_import void mtVectorTMulMatrix(MtVector *pVecX, MtMatrix *pMatY, MtVector *pVecZ, size_t stream, int *pCode);

mt_import void mtNewVectorAccBuffer(MtVector *pVec, MtBuffer *pBuf, int *pCode);
mt_import void mtVectorSum(MtVector *pVecX, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVectorSquareSum(MtVector *pVecX, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVectorMin(MtVector *pVecX, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVectorMax(MtVector *pVecX, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVectorDot(MtVector *pVecX, MtVector *pVecY, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVectorSumSquareSum(MtVector *pVecX, MtVector *pVecY, MtBuffer buf, size_t stream, float *pRes, int *pCode);
mt_import void mtVectorDiffSquareSum(MtVector *pVecX, MtVector *pVecY, MtBuffer buf, size_t stream, float *pRes, int *pCode);

mt_import void mtNewMatrix(size_t w, size_t h, MtMatrix **ppMat, float **ppBuf, int *pCode);
mt_import void mtMatrixDestroy(MtMatrix *pMat);

mt_import void mtMatrixT(MtMatrix *pMatX, MtMatrix *pMatY, cudaStream_t stream, int *pCode);
mt_import void mtMatrixAddScalar(MtMatrix *pMatX, MtMatrix *pMatY, float *pA, size_t stream, int *pCode);
mt_import void mtMatrixMulScalar(MtMatrix *pMatX, MtMatrix *pMatY, float *pA, size_t stream, int *pCode);
mt_import void mtMatrixMulAddScalar(MtMatrix *pMatX, MtMatrix *pMatY, float *pA, float *pB, size_t stream, int *pCode);
mt_import void mtMatrixAddMatrix(MtMatrix *pMatX, MtMatrix *pMatY, MtMatrix *pMatZ, size_t stream, int *pCode);
mt_import void mtMatrixSubMatrix(MtMatrix *pMatX, MtMatrix *pMatY, MtMatrix *pMatZ, size_t stream, int *pCode);
mt_import void mtMatrixMulMatrix(MtMatrix *pMatX, MtMatrix *pMatY, MtMatrix *pMatZ, size_t stream, int *pCode);
mt_import void mtMatrixMulVector(MtMatrix *pMatX, MtVector *pVecY, MtVector *pVecZ, size_t stream, int *pCode);

mt_import void mtNewMatrixWideBuffer(MtMatrix *pMat, MtBuffer *pBuf, int *pCode);
mt_import void mtMatrixWideMulVector(MtMatrix *pMatX, MtVector *pVecY, MtVector *pVecZ, size_t stream, int *pCode);
