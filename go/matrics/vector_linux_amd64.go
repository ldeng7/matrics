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

func (tx *Tensor) mtVecAddScalar(ty *Tensor, a float32, stream Stream, code *int) {
	C.mtVecAddScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecMulScalar(ty *Tensor, a float32, stream Stream, code *int) {
	C.mtVecMulScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecMulAddScalar(ty *Tensor, a, b float32, stream Stream, code *int) {
	C.mtVecMulAddScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)), (*C.float)(unsafe.Pointer(&b)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecPowScalar(ty *Tensor, a float32, stream Stream, code *int) {
	C.mtVecPowScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecPowMulScalar(ty *Tensor, a, b float32, stream Stream, code *int) {
	C.mtVecPowMulScalar(tx.handle, ty.handle, (*C.float)(unsafe.Pointer(&a)), (*C.float)(unsafe.Pointer(&b)),
		C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecAddVec(ty, tz *Tensor, stream Stream, code *int) {
	C.mtVecAddVec(tx.handle, ty.handle, tz.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecSubVec(ty, tz *Tensor, stream Stream, code *int) {
	C.mtVecSubVec(tx.handle, ty.handle, tz.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecPatchMulVec(ty, tz *Tensor, stream Stream, code *int) {
	C.mtVecPatchMulVec(tx.handle, ty.handle, tz.handle, C.MtStream(stream), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtNewVecAccBuffer(buf *Buffer, code *int) {
	C.mtNewVecAccBuffer(tx.handle, &buf.handle, (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecSum(buf *Buffer, stream Stream, res *float32, code *int) {
	C.mtVecSum(tx.handle, buf.handle,
		C.MtStream(stream), (*C.float)(unsafe.Pointer(res)), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecSquareSum(buf *Buffer, stream Stream, res *float32, code *int) {
	C.mtVecSquareSum(tx.handle, buf.handle,
		C.MtStream(stream), (*C.float)(unsafe.Pointer(res)), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecMin(buf *Buffer, stream Stream, res *float32, code *int) {
	C.mtVecMin(tx.handle, buf.handle,
		C.MtStream(stream), (*C.float)(unsafe.Pointer(res)), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecMax(buf *Buffer, stream Stream, res *float32, code *int) {
	C.mtVecMax(tx.handle, buf.handle,
		C.MtStream(stream), (*C.float)(unsafe.Pointer(res)), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecDot(ty *Tensor, buf *Buffer, stream Stream, res *float32, code *int) {
	C.mtVecDot(tx.handle, ty.handle, buf.handle,
		C.MtStream(stream), (*C.float)(unsafe.Pointer(res)), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecSumSquareSum(ty *Tensor, buf *Buffer, stream Stream, res *float32, code *int) {
	C.mtVecSumSquareSum(tx.handle, ty.handle, buf.handle,
		C.MtStream(stream), (*C.float)(unsafe.Pointer(res)), (*C.int)(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecDiffSquareSum(ty *Tensor, buf *Buffer, stream Stream, res *float32, code *int) {
	C.mtVecDiffSquareSum(tx.handle, ty.handle, buf.handle,
		C.MtStream(stream), (*C.float)(unsafe.Pointer(res)), (*C.int)(unsafe.Pointer(code)))
}
