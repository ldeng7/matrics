#include "internal.h"
#include <stdio.h>

const uint32 MAT_NTB_X = 32;
const uint32 MAT_NTB_Y = 32;

#define activate(e, act) \
    switch (act.typ) { \
    case MtNnActivationTypeReLU: e = (e < 0) ? 0 : e; break; \
    case MtNnActivationTypeLReLU: e = (e < 0) ? e * act.a : e; break; \
    case MtNnActivationTypeELU: e = (e < 0) ? (__expf(e) - 1) * act.a : 0; break; \
    case MtNnActivationTypeSwish: e = (e < 0) ? e / (1 + __expf(-e * act.a)) : 0; break; \
    }

__global__ void devMatMulMatAddVecTAct(float *x, float *y, float *z, float *w, uint32 m, uint32 n, uint32 l,
        MtNnActivation act) {
    uint32 i = blockIdx.x * MAT_NTB_X + threadIdx.x;
    uint32 j = blockIdx.y * MAT_NTB_Y + threadIdx.y;
    if (i >= l || j >= m) return;
    float s = 0;
    uint32 nj = n * j;
    for (uint32 k = 0; k < n; k++) {
        s += x[nj + k] * y[l * k + i];
    }
    s += z[i];
    activate(s, act);
    w[l * j + i] = s;
}

__export void mtMatMulMatAddVecTAct(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtNnActivation *act,
        cudaStream_t stream, int *pCode) {
    uint32 nbx = divCeil(pY->nx, MAT_NTB_X);
    uint32 nby = divCeil(pX->ny, MAT_NTB_Y);
    cudaGetLastError();
    devMatMulMatAddVecTAct<<<dim3(nbx, nby, 1), dim3(MAT_NTB_X, MAT_NTB_Y, 1), 0, stream>>>(
        pX->p, pY->p, pZ->p, pW->p, pX->ny, pX->nx, pY->nx, *act);
    syncCheck(stream, pCode);
}

__global__ void devVecTMulMatAddVecTAct(float *x, float *y, float *z, float *w, uint32 n, uint32 l,
        MtNnActivation act) {
    uint32 i = blockIdx.x * MAT_NTB_X + threadIdx.x;
    if (i >= l) return;
    float s = 0;
    for (uint32 k = 0; k < n; k++) {
        s += x[k] * y[l * k + i];
    }
    s += z[i];
    activate(s, act);
    w[i] = s;
}

__export void mtVecTMulMatAddVecTAct(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtNnActivation *act,
        cudaStream_t stream, int *pCode) {
    cudaGetLastError();
    devVecTMulMatAddVecTAct<<<divCeil(pY->nx, MAT_NTB_X), MAT_NTB_X, 0, stream>>>(
        pX->p, pY->p, pZ->p, pW->p, pX->nx, pY->nx, *act);
    syncCheck(stream, pCode);
}

// szi/szo: {x: width, y: height, z: depth}
// szc: {x: width, y: height}
__global__ void devCubConv2dSame(float *x, float *y, float *z, float *w, uint3 szi, uint2 szc, uint3 szo,
        MtConv2dConf conf) {
    int32 io = blockIdx.x * MAT_NTB_X + threadIdx.x;
    int32 jo = blockIdx.y * MAT_NTB_Y + threadIdx.y;
    int32 ko = blockIdx.z;
    if (io >= szo.x || jo >= szo.y) return;

    int32 icb = (szo.x - 1) * conf.cnn.strideX + szc.x - szi.x;
    int32 jcb = (szo.y - 1) * conf.cnn.strideY + szc.y - szi.y;
    icb = icb >= 0 ? icb >> 1 : 0;
    jcb = jcb >= 0 ? jcb >> 1 : 0;
    int32 ib = io * conf.cnn.strideX - icb;
    int32 jb = jo * conf.cnn.strideY - jcb;
    int32 ie = ib + szc.x;
    int32 je = jb + szc.y;
    ib = ib >= 0 ? (icb = 0, ib) : (icb = -ib, 0);
    jb = jb >= 0 ? (jcb = 0, jb) : (jcb = -jb, 0);
    ie = ie <= szi.x ? ie : szi.x;
    je = je <= szi.y ? je : szi.y;

    float s = 0;
    int32 pcko = szc.x * szc.y * szi.z * ko;
    for (int32 k = 0; k < szi.z; k++) {
        int32 pk = szi.x * szi.y * k;
        int32 pck = pcko + szc.x * szc.y * k;
        for (int32 j = jb, jc = jcb; j < je; j++, jc++) {
            int32 pj = pk + szi.x * j;
            int32 pcj = pck + szc.x * jc;
            for (int32 i = ib, ic = icb; i < ie; i++, ic++) {
                s += x[pj + i] * y[pcj + ic];
            }
        }
    }
    s += z[ko];
    activate(s, conf.act);
    w[szo.x * szo.y * ko + szo.x * jo + io] = s;
}

__global__ void devCubConv2dValid(float *x, float *y, float *z, float *w, uint3 szi, uint2 szc, uint3 szo,
        MtConv2dConf conf) {
    uint32 io = blockIdx.x * MAT_NTB_X + threadIdx.x;
    uint32 jo = blockIdx.y * MAT_NTB_Y + threadIdx.y;
    uint32 ko = blockIdx.z;
    if (io >= szo.x || jo >= szo.y) return;

    uint32 ib = io * conf.cnn.strideX;
    uint32 jb = jo * conf.cnn.strideY;
    uint32 ie = ib + szc.x;
    uint32 je = jb + szc.y;

    float s = 0;
    uint32 pcko = szc.x * szc.y * szi.z * ko;
    for (uint32 k = 0; k < szi.z; k++) {
        uint32 pk = szi.x * szi.y * k;
        uint32 pck = pcko + szc.x * szc.y * k;
        for (uint32 j = jb, jc = 0; j < je; j++, jc++) {
            uint32 pj = pk + szi.x * j;
            uint32 pcj = pck + szc.x * jc;
            for (uint32 i = ib, ic = 0; i < ie; i++, ic++) {
                s += x[pj + i] * y[pcj + ic];
            }
        }
    }
    s += z[ko];
    activate(s, conf.act);
    w[szo.x * szo.y * ko + szo.x * jo + io] = s;
}

__export void mtCubBatchConv2d(MtTensor *pX, MtTensor *pY, MtTensor *pZ, MtTensor *pW, MtConv2dConf *conf,
        cudaStream_t stream, int *pCode) {
    float *px = pX->p, *pw = pW->p;
    uint3 szi = make_uint3(pX->nx, pX->ny, pX->nz);
    uint2 szc = make_uint2(pY->nx, pY->ny);
    uint3 szo = make_uint3(pW->nx, pW->ny, pW->nz);
    uint32 strideX = szi.x * szi.y * szi.z;
    uint32 strideW = szo.x * szo.y * szo.z;
    uint32 nbx = divCeil(szo.x, MAT_NTB_X);
    uint32 nby = divCeil(szo.y, MAT_NTB_Y);

    auto f = devCubConv2dSame;
    switch (conf->cnn.paddingTyp) {
    case MtCnnPaddingTypeValid: f = devCubConv2dValid; break;
    }
    cudaGetLastError();
    for (uint32 i = 0; i < pX->nw; i++, px += strideX, pw += strideW) {
        f<<<dim3(nbx, nby, szo.z), dim3(MAT_NTB_X, MAT_NTB_Y, 1), 0, stream>>>(
            px, pY->p, pZ->p, pw, szi, szc, szo, *conf);
    }
    syncCheck(stream, pCode);
}

#define devCubPool2dSame(rInit, rExp, rFinish) \
    int32 io = blockIdx.x * MAT_NTB_X + threadIdx.x; \
    int32 jo = blockIdx.y * MAT_NTB_Y + threadIdx.y; \
    int32 k = blockIdx.z; \
    if (io >= szo.x || jo >= szo.y) return; \
    int32 icb = (szo.x - 1) * conf.cnn.strideX + conf.coreX - szi.x; \
    int32 jcb = (szo.y - 1) * conf.cnn.strideY + conf.coreY - szi.y; \
    icb = icb >= 0 ? icb >> 1 : 0; \
    jcb = jcb >= 0 ? jcb >> 1 : 0; \
    int32 ib = io * conf.cnn.strideX - icb; \
    int32 jb = jo * conf.cnn.strideY - jcb; \
    int32 ie = ib + conf.coreX; \
    int32 je = jb + conf.coreY; \
    ib = ib >= 0 ? ib : 0; \
    jb = jb >= 0 ? jb : 0; \
    ie = ie <= szi.x ? ie : szi.x; \
    je = je <= szi.y ? je : szi.y; \
    float r = (rInit); \
       int32 pk = szi.x * szi.y * k; \
    for (int32 j = jb; j < je; j++) { \
        int32 pj = pk + szi.x * j; \
        for (int32 i = ib; i < ie; i++) { \
            float e = x[pj + i]; \
            rExp; \
        } \
    } \
    rFinish; \
    y[szo.x * szo.y * k + szo.x * jo + io] = r;

#define devCubPool2dValid(rInit, rExp, rFinish) \
    uint32 io = blockIdx.x * MAT_NTB_X + threadIdx.x; \
    uint32 jo = blockIdx.y * MAT_NTB_Y + threadIdx.y; \
    uint32 k = blockIdx.z; \
    if (io >= szo.x || jo >= szo.y) return; \
    uint32 ib = io * conf.cnn.strideX; \
    uint32 jb = jo * conf.cnn.strideY; \
    uint32 ie = ib + conf.coreX; \
    uint32 je = jb + conf.coreY; \
    float r = (rInit); \
       uint32 pk = szi.x * szi.y * k; \
    for (uint32 j = jb; j < je; j++) { \
        uint32 pj = pk + szi.x * j; \
        for (uint32 i = ib; i < ie; i++) { \
            float e = x[pj + i]; \
            rExp; \
        } \
    } \
    rFinish; \
    y[szo.x * szo.y * k + szo.x * jo + io] = r;

__global__ void devCubPool2dSameMax(float *x, float *y, uint3 szi, uint2 szo, MtPool2dConf conf) {
    devCubPool2dSame(NAN, r = fmaxf(r, e), r += 0);
}

__global__ void devCubPool2dSameAvg(float *x, float *y, uint3 szi, uint2 szo, MtPool2dConf conf) {
    devCubPool2dSame(0, r += e, r /= (ie - ib) * (je - jb));
}

__global__ void devCubPool2dValidMax(float *x, float *y, uint3 szi, uint2 szo, MtPool2dConf conf) {
    devCubPool2dValid(NAN, r = fmaxf(r, e), r += 0);
}

__global__ void devCubPool2dValidAvg(float *x, float *y, uint3 szi, uint2 szo, MtPool2dConf conf) {
    devCubPool2dValid(0, r += e, r /= (ie - ib) * (je - jb));
}

__export void mtCubBatchPool2d(MtTensor *pX, MtTensor *pY, MtPool2dConf *conf, cudaStream_t stream, int *pCode) {
    float *px = pX->p, *py = pY->p;
    uint3 szi = make_uint3(pX->nx, pX->ny, pX->nz);
    uint2 szo = make_uint2(pY->nx, pY->ny);
    uint32 strideX = szi.x * szi.y * szi.z;
    uint32 strideY = szo.x * szo.y * szi.z;
    uint32 nbx = divCeil(szo.x, MAT_NTB_X);
    uint32 nby = divCeil(szo.y, MAT_NTB_Y);

    auto f = devCubPool2dSameMax;
    switch (conf->cnn.paddingTyp) {
    case MtCnnPaddingTypeSame:
        switch (conf->typ) {
        case MtPoolTypeAvg: f = devCubPool2dSameAvg; break;
        } break;
    case MtCnnPaddingTypeValid:
        switch (conf->typ) {
        case MtPoolTypeMax: f = devCubPool2dValidMax; break;
        case MtPoolTypeAvg: f = devCubPool2dValidAvg; break;
        } break;
    }
    cudaGetLastError();
    for (uint32 i = 0; i < pX->nw; i++, px += strideX, py += strideY) {
        f<<<dim3(nbx, nby, pY->nz), dim3(MAT_NTB_X, MAT_NTB_Y, 1), 0, stream>>>(
            px, py, szi, szo, *conf);
    }
    syncCheck(stream, pCode);
}
