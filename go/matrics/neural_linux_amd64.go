package matrics

/*
#cgo CFLAGS: -I ../../libmatrics/include
#cgo LDFLAGS: -lmatrics
#include "libmatrics.h"
*/
import "C"
import (
	"unsafe"
)

func (tx *Tensor) mtMatMulMatAddVecTAct(ty, tz, tw *Tensor, act *Activation, stream Stream, code *int) {
	C.mtMatMulMatAddVecTAct(tx.handle, ty.handle, tz.handle, tw.handle, (*C.MtNnActivation)(unsafe.Pointer(act)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecTMulMatAddVecTAct(ty, tz, tw *Tensor, act *Activation, stream Stream, code *int) {
	C.mtVecTMulMatAddVecTAct(tx.handle, ty.handle, tz.handle, tw.handle, (*C.MtNnActivation)(unsafe.Pointer(act)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtCubBatchConv2d(ty, tz, tw *Tensor, conf *Conv2dConf, stream Stream, code *int) {
	C.mtCubBatchConv2d(tx.handle, ty.handle, tz.handle, tw.handle, (*C.MtConv2dConf)(unsafe.Pointer(conf)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtCubBatchPool2d(ty *Tensor, conf *Pool2dConf, stream Stream, code *int) {
	C.mtCubBatchPool2d(tx.handle, ty.handle, (*C.MtPool2dConf)(unsafe.Pointer(conf)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}
