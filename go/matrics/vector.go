package matrics

import (
	"unsafe"
)

// ind

func (tx *Tensor) vecInd1s1(f mtFunc, ty *Tensor, a float32, stream Stream) error {
	if tx.nx != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	f.Call(tx.handler, ty.handler, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) vecInd1s2(f mtFunc, ty *Tensor, a, b float32, stream Stream) error {
	if tx.nx != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	f.Call(tx.handler, ty.handler, uintptr(unsafe.Pointer(&a)), uintptr(unsafe.Pointer(&b)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) vecInd2(f mtFunc, ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	f.Call(tx.handler, ty.handler, tz.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) VecAddScalar(ty *Tensor, a float32, stream Stream) error {
	return tx.vecInd1s1(mtVecAddScalar, ty, a, stream)
}

func (tx *Tensor) VecMulScalar(ty *Tensor, a float32, stream Stream) error {
	return tx.vecInd1s1(mtVecMulScalar, ty, a, stream)
}

func (tx *Tensor) VecMulAddScalar(ty *Tensor, a, b float32, stream Stream) error {
	return tx.vecInd1s2(mtVecMulAddScalar, ty, a, b, stream)
}

func (tx *Tensor) VecPowScalar(ty *Tensor, a float32, stream Stream) error {
	return tx.vecInd1s1(mtVecPowScalar, ty, a, stream)
}

func (tx *Tensor) VecPowMulScalar(ty *Tensor, a, b float32, stream Stream) error {
	return tx.vecInd1s2(mtVecPowMulScalar, ty, a, b, stream)
}

func (tx *Tensor) VecAddVec(ty, tz *Tensor, stream Stream) error {
	return tx.vecInd2(mtVecAddVec, ty, tz, stream)
}

func (tx *Tensor) VecSubVec(ty, tz *Tensor, stream Stream) error {
	return tx.vecInd2(mtVecSubVec, ty, tz, stream)
}

func (tx *Tensor) VecPatchMulVec(ty, tz *Tensor, stream Stream) error {
	return tx.vecInd2(mtVecPatchMulVec, ty, tz, stream)
}

// acc

type VecAccBuffer struct {
	handler uintptr
	vecNx   uint64
}

func (tx *Tensor) NewAccBuffer() (*VecAccBuffer, error) {
	buf := &VecAccBuffer{vecNx: tx.nx}
	var code int
	mtNewVecAccBuffer.Call(tx.handler,
		uintptr(unsafe.Pointer(&buf.handler)), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return nil, &mtError{code, "failed to alloc"}
	}
	return buf, nil
}

func (buf *VecAccBuffer) Destroy() {
	mtBufferDestroy.Call(buf.handler)
}

func (tx *Tensor) vecAcc(f mtFunc, buf *VecAccBuffer, stream Stream) (float32, error) {
	if tx.nx != buf.vecNx {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	f.Call(tx.handler, buf.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&res)), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}

func (tx *Tensor) vecAcc1(f mtFunc, ty *Tensor, buf *VecAccBuffer, stream Stream) (float32, error) {
	if tx.nx != ty.nx || tx.nx != buf.vecNx {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	f.Call(tx.handler, ty.handler, buf.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&res)), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}

func (tx *Tensor) VecSum(buf *VecAccBuffer, stream Stream) (float32, error) {
	return tx.vecAcc(mtVecSum, buf, stream)
}

func (tx *Tensor) VecSquareSum(buf *VecAccBuffer, stream Stream) (float32, error) {
	return tx.vecAcc(mtVecSquareSum, buf, stream)
}

func (tx *Tensor) VecMin(buf *VecAccBuffer, stream Stream) (float32, error) {
	return tx.vecAcc(mtVecMin, buf, stream)
}

func (tx *Tensor) VecMax(buf *VecAccBuffer, stream Stream) (float32, error) {
	return tx.vecAcc(mtVecMax, buf, stream)
}

func (tx *Tensor) VecDot(ty *Tensor, buf *VecAccBuffer, stream Stream) (float32, error) {
	return tx.vecAcc1(mtVecDot, ty, buf, stream)
}

func (tx *Tensor) VecSumSquareSum(ty *Tensor, buf *VecAccBuffer, stream Stream) (float32, error) {
	return tx.vecAcc1(mtVecSumSquareSum, ty, buf, stream)
}

func (tx *Tensor) VecDiffSquareSum(ty *Tensor, buf *VecAccBuffer, stream Stream) (float32, error) {
	return tx.vecAcc1(mtVecDiffSquareSum, ty, buf, stream)
}
