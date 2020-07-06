#include "internal.h"

const size_t CUB_NTB_X = 32;
const size_t CUB_NTB_Y = 32;

// sz: {x: depth, y: width, z: height}
// szc: {x: depth_out, y: width, z: height}
// szo: {x: _, y: width, z: height}
__global__ void devCubConv2d(float *x, float *y, float *z, float *w, ulonglong3 sz, ulonglong3 szc, ulonglong3 szo,
        MtConv2dConf conf) {
    size_t io = size_t(blockIdx.x) * CUB_NTB_X + size_t(threadIdx.x);
    size_t jo = size_t(blockIdx.y) * CUB_NTB_Y + size_t(threadIdx.y);
    size_t ko = size_t(blockIdx.z);
    if (io >= szo.y || jo >= szo.z) return;
    size_t i = io * conf.cnn.strideY;
    size_t j = jo * conf.cnn.strideZ;
    size_t ie = i + szc.y;
    size_t je = j + szc.z;
    ie = ie < sz.y ? ie : sz.y;
    je = je < sz.z ? je : sz.z;

    float s = 0;
    size_t pc0 = sz.x * szc.y * szc.z * ko;
    for (size_t jc = 0; j < je; j++, jc++) {
    	size_t p1 = sz.x * sz.y * j;
		size_t pc1 = pc0 + sz.x * szc.y * jc;
    	for (size_t ic = 0; i < ie; i++, ic++) {
			size_t p = p1 + sz.x * i;
			size_t pc = pc1 + sz.x + ic;
			for (size_t k = 0; k < sz.x; k++) {
				s += x[p] * y[pc];
			}
		}
    }
    s += z[ko];
    activate(s, conf.act.typ, conf.act.a);
    w[szc.x * szo.y * jo + szc.x * io + ko] = s;
}

__export void mtCubConv2d(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtConv2dConf *conf,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    size_t nbx = divCeil(pW->ny, CUB_NTB_X);
    size_t nby = divCeil(pW->nz, CUB_NTB_Y);
    ulonglong3 sz = make_ulonglong3(pX->nx, pX->ny, pX->nz);
    ulonglong3 szc = make_ulonglong3(pY->nw, pY->ny, pY->nz);
    ulonglong3 szo = make_ulonglong3(pW->nx, pW->ny, pW->nz);
    devCubConv2d<<<dim3(nbx, nby, pY->nw), dim3(CUB_NTB_X, CUB_NTB_Y, 1), 0, stream>>>(
        pX->p, pY->p, pZ->p, pW->p, sz, szc, szo, *conf);
    syncCheck;
}

__export void mtTesConv2d(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtConv2dConf *conf,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    size_t nbx = divCeil(pW->ny, CUB_NTB_X);
    size_t nby = divCeil(pW->nz, CUB_NTB_Y);
    float *px = pX->p;
    float *pw = pW->p;
    size_t strideX = pX->nx * pX->ny * pX->nz;
    size_t strideW = pW->nx * pW->ny * pW->nz;
    ulonglong3 sz = make_ulonglong3(pX->nx, pX->ny, pX->nz);
    ulonglong3 szc = make_ulonglong3(pY->nw, pY->ny, pY->nz);
    ulonglong3 szo = make_ulonglong3(pW->nx, pW->ny, pW->nz);
    for (size_t i = 0; i < pX->nw; i++, px += strideX, pw += strideW) {
        devCubConv2d<<<dim3(nbx, nby, pY->nw), dim3(CUB_NTB_X, CUB_NTB_Y, 1), 0, stream>>>(
            px, pY->p, pZ->p, pw, sz, szc, szo, *conf);
    }
    syncCheck;
}

// sz: {x: depth, y: width, z: height}
// szo: {x: _, y: width, z: height}
__global__ void devCubCnnPoolMax(float *x, float *y, ulonglong3 sz, ulonglong3 szo, MtCnnPoolConf conf) {
}

__global__ void devCubCnnPoolAvg(float *x, float *y, ulonglong3 sz, ulonglong3 szo, MtCnnPoolConf conf) {
}

__export void mtCubCnnPool(MtTensor *pX, MtTensor *pY, MtCnnPoolConf *conf, cudaStream_t stream, int *pCode) {
}

__export void mtTesCnnPool(MtTensor *pX, MtTensor *pY, MtCnnPoolConf *conf, cudaStream_t stream, int *pCode) {
}
