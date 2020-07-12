package matrics

import (
	"unsafe"
)

func (tx *Tensor) mtMatT(ty *Tensor, stream Stream, code *int) {
	mtCMatT.Call(tx.handle, ty.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatAddScalar(ty *Tensor, a float32, stream Stream, code *int) {
	mtCMatAddScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatMulScalar(ty *Tensor, a float32, stream Stream, code *int) {
	mtCMatMulScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatMulAddScalar(ty *Tensor, a, b float32, stream Stream, code *int) {
	mtCMatMulAddScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)), uintptr(unsafe.Pointer(&b)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatPowScalar(ty *Tensor, a float32, stream Stream, code *int) {
	mtCMatPowScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatPowMulScalar(ty *Tensor, a, b float32, stream Stream, code *int) {
	mtCMatPowMulScalar.Call(tx.handle, ty.handle, uintptr(unsafe.Pointer(&a)), uintptr(unsafe.Pointer(&b)),
		uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatAddMat(ty, tz *Tensor, stream Stream, code *int) {
	mtCMatAddMat.Call(tx.handle, ty.handle, tz.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatSubMat(ty, tz *Tensor, stream Stream, code *int) {
	mtCMatSubMat.Call(tx.handle, ty.handle, tz.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatMulMat(ty, tz *Tensor, stream Stream, code *int) {
	mtCMatMulMat.Call(tx.handle, ty.handle, tz.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtVecTMulMat(ty, tz *Tensor, stream Stream, code *int) {
	mtCVecTMulMat.Call(tx.handle, ty.handle, tz.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}

func (tx *Tensor) mtMatMulVec(ty, tz *Tensor, stream Stream, code *int) {
	mtCMatMulVec.Call(tx.handle, ty.handle, tz.handle, uintptr(stream), uintptr(unsafe.Pointer(code)))
}
