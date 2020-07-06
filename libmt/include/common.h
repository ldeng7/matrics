#pragma once

typedef struct {
    float *p;
    size_t nx, ny, nz, nw;
} MtTensor;

typedef float *MtBuffer;

typedef unsigned long MtActivationType;
const MtActivationType MtActivationTypeNone  = 0;
const MtActivationType MtActivationTypeReLU  = 1;
const MtActivationType MtActivationTypeLReLU = 2;
const MtActivationType MtActivationTypeELU   = 3;
const MtActivationType MtActivationTypeSwish = 4;

typedef struct {
    MtActivationType typ;
    float a;
} MtActivation;

typedef struct {
    unsigned char strideY;
    unsigned char strideZ;
} MtCnnConf;

typedef struct {
    MtActivation act;
    MtCnnConf cnn;
} MtConv2dConf;

typedef unsigned char MtCnnPoolType;
const MtCnnPoolType MtCnnPoolTypeMax = 0;
const MtCnnPoolType MtCnnPoolTypeAvg = 1;

typedef struct {
    MtCnnPoolType typ;
    unsigned char coreY;
    unsigned char coreZ;
    MtCnnConf cnn;
} MtCnnPoolConf;
