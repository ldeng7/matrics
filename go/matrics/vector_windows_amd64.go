package matrics

import (
	"unsafe"
)

func (tx *Tensor) mtVecAddScalar(ty *Tensor, a float32, stream Stream, code *int) {
	mtCVecAddScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecMulScalar(ty *Tensor, a float32, stream Stream, code *int) {
	mtCVecMulScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecMulAddScalar(ty *Tensor, a, b float32, stream Stream, code *int) {
	mtCVecMulAddScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)), uintptr(unsafe.Pointer(&b)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecPowScalar(ty *Tensor, a float32, stream Stream, code *int) {
	mtCVecPowScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecPowMulScalar(ty *Tensor, a, b float32, stream Stream, code *int) {
	mtCVecPowMulScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)), uintptr(unsafe.Pointer(&b)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecAddVec(ty, tz *Tensor, stream Stream, code *int) {
	mtCVecAddVec.Call(tx.handle, ty.handle, tz.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecSubVec(ty, tz *Tensor, stream Stream, code *int) {
	mtCVecSubVec.Call(tx.handle, ty.handle, tz.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecPatchMulVec(ty, tz *Tensor, stream Stream, code *int) {
	mtCVecPatchMulVec.Call(tx.handle, ty.handle, tz.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtNewVecAccBuffer(buf *Buffer, code *int) {
	mtCNewVecAccBuffer.Call(tx.handle, uintptr(unsafe.Pointer(&buf.handle)), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecSum(buf *Buffer, stream Stream, res *float32, code *int) {
	mtCVecSum.Call(tx.handle, buf.handle,
		uintptr(stream), uintptr(unsafe.Pointer(res)), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecSquareSum(buf *Buffer, stream Stream, res *float32, code *int) {
	mtCVecSquareSum.Call(tx.handle, buf.handle,
		uintptr(stream), uintptr(unsafe.Pointer(res)), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecMin(buf *Buffer, stream Stream, res *float32, code *int) {
	mtCVecMin.Call(tx.handle, buf.handle,
		uintptr(stream), uintptr(unsafe.Pointer(res)), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecMax(buf *Buffer, stream Stream, res *float32, code *int) {
	mtCVecMax.Call(tx.handle, buf.handle,
		uintptr(stream), uintptr(unsafe.Pointer(res)), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecDot(ty *Tensor, buf *Buffer, stream Stream, res *float32, code *int) {
	mtCVecDot.Call(tx.handle, ty.handle, buf.handle,
		uintptr(stream), uintptr(unsafe.Pointer(res)), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecSumSquareSum(ty *Tensor, buf *Buffer, stream Stream, res *float32, code *int) {
	mtCVecSumSquareSum.Call(tx.handle, ty.handle, buf.handle,
		uintptr(stream), uintptr(unsafe.Pointer(res)), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecDiffSquareSum(ty *Tensor, buf *Buffer, stream Stream, res *float32, code *int) {
	mtCVecDiffSquareSum.Call(tx.handle, ty.handle, buf.handle,
		uintptr(stream), uintptr(unsafe.Pointer(res)), uintptr(unsafe.Pointer(code)))
}
