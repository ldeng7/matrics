#pragma once

typedef unsigned char uint8;
typedef int int32;
typedef unsigned int uint32;

typedef struct {
    float *p;
    uint32 n;
    uint32 nx, ny, nz, nw;
} MtTensor;

typedef size_t MtStream;
typedef size_t MtBuffer;

typedef uint32 MtNnActivationType;
#define MtNnActivationTypeNone  0
#define MtNnActivationTypeReLU  1
#define MtNnActivationTypeLReLU 2
#define MtNnActivationTypeELU   3
#define MtNnActivationTypeSwish 4

typedef struct {
    MtNnActivationType typ;
    float a;
} MtNnActivation;

typedef uint8 MtCnnPaddingType;
#define MtCnnPaddingTypeSame  0
#define MtCnnPaddingTypeValid 1
typedef uint8 MtPoolType;
#define MtPoolTypeMax 0
#define MtPoolTypeAvg 1

typedef struct {
    MtCnnPaddingType paddingTyp;
    uint8 strideX;
    uint8 strideY;
} MtCnn2dConf;

typedef struct {
    MtNnActivation act;
    MtCnn2dConf cnn;
} MtConv2dConf;

typedef struct {
    MtPoolType typ;
    uint8 coreX;
    uint8 coreY;
    MtCnn2dConf cnn;
} MtPool2dConf;
