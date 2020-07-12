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

func (tx *Tensor) mtMatT(ty *Tensor, stream Stream, code *int) {
	C.mtMatT(tx.handle, ty.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatAddScalar(ty *Tensor, a float32, stream Stream, code *int) {
	C.mtMatAddScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatMulScalar(ty *Tensor, a float32, stream Stream, code *int) {
	C.mtMatMulScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatMulAddScalar(ty *Tensor, a, b float32, stream Stream, code *int) {
	C.mtMatMulAddScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)), (*C.float)(unsafe.Pointer(&b)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatPowScalar(ty *Tensor, a float32, stream Stream, code *int) {
	C.mtMatPowScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatPowMulScalar(ty *Tensor, a, b float32, stream Stream, code *int) {
	C.mtMatPowMulScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)), (*C.float)(unsafe.Pointer(&b)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatAddMat(ty, tz *Tensor, stream Stream, code *int) {
	C.mtMatAddMat(tx.handle, ty.handle, tz.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatSubMat(ty, tz *Tensor, stream Stream, code *int) {
	C.mtMatSubMat(tx.handle, ty.handle, tz.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatMulMat(ty, tz *Tensor, stream Stream, code *int) {
	C.mtMatMulMat(tx.handle, ty.handle, tz.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecTMulMat(ty, tz *Tensor, stream Stream, code *int) {
	C.mtVecTMulMat(tx.handle, ty.handle, tz.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatMulVec(ty, tz *Tensor, stream Stream, code *int) {
	C.mtMatMulVec(tx.handle, ty.handle, tz.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}
