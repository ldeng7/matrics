#pragma once

typedef struct MtVector {
    float *p;
    size_t n;
} MtVector;

typedef struct MtMatrix {
    float *p;
    size_t w, h;
} MtMatrix;

typedef float *MtBuffer;
