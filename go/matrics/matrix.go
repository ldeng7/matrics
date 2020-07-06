package matrics

import (
	"unsafe"
)

// ind

func (tx *Tensor) MatT(ty *Tensor, stream Stream) error {
	if tx.nx != ty.ny || tx.ny != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	mtMatT.Call(tx.handler, ty.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) matInd1s1(f mtFunc, ty *Tensor, a float32, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny {
		return unsuitableShapeErr
	}
	var code int
	f.Call(tx.handler, ty.handler, uintptr(unsafe.Pointer(&a)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) matInd1s2(f mtFunc, ty *Tensor, a, b float32, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny {
		return unsuitableShapeErr
	}
	var code int
	f.Call(tx.handler, ty.handler, uintptr(unsafe.Pointer(&a)), uintptr(unsafe.Pointer(&b)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) matInd2(f mtFunc, ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny ||
		tx.nx != tz.nx || tx.ny != tz.ny {
		return unsuitableShapeErr
	}
	var code int
	f.Call(tx.handler, ty.handler, tz.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) MatAddScalar(ty *Tensor, a float32, stream Stream) error {
	return tx.matInd1s1(mtMatAddScalar, ty, a, stream)
}

func (tx *Tensor) MatMulScalar(ty *Tensor, a float32, stream Stream) error {
	return tx.matInd1s1(mtMatMulScalar, ty, a, stream)
}

func (tx *Tensor) MatMulAddScalar(ty *Tensor, a, b float32, stream Stream) error {
	return tx.matInd1s2(mtMatMulAddScalar, ty, a, b, stream)
}

func (tx *Tensor) MatPowScalar(ty *Tensor, a float32, stream Stream) error {
	return tx.matInd1s1(mtMatPowScalar, ty, a, stream)
}

func (tx *Tensor) MatPowMulScalar(ty *Tensor, a, b float32, stream Stream) error {
	return tx.matInd1s2(mtMatPowMulScalar, ty, a, b, stream)
}

func (tx *Tensor) MatAddMat(ty, tz *Tensor, stream Stream) error {
	return tx.matInd2(mtMatAddMat, ty, tz, stream)
}

func (tx *Tensor) MatSubMat(ty, tz *Tensor, stream Stream) error {
	return tx.matInd2(mtMatSubMat, ty, tz, stream)
}

// matmul

func (tx *Tensor) MatMulMat(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.ny || tx.ny != tz.ny || ty.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	mtMatMulMat.Call(tx.handler, ty.handler, tz.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) VecTMulMat(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.ny || ty.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	mtVecTMulMat.Call(tx.handler, ty.handler, tz.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) MatMulVec(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	mtMatMulVec.Call(tx.handler, ty.handler, tz.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) MatMulMatAddVecTAct(ty, tz, tw *Tensor, act *Activation, stream Stream) error {
	if tx.nx != ty.ny || tx.ny != tw.ny || ty.nx != tw.nx || ty.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	mtMatMulMatAddVecTAct.Call(tx.handler, ty.handler, tz.handler, tw.handler, uintptr(unsafe.Pointer(act)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

func (tx *Tensor) VecTMulMatAddVecTAct(ty, tz, tw *Tensor, act *Activation, stream Stream) error {
	if tx.nx != ty.ny || ty.nx != tw.nx || ty.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	mtVecTMulMatAddVecTAct.Call(tx.handler, ty.handler, tz.handler, tw.handler, uintptr(unsafe.Pointer(act)),
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}

type MatFlatBuffer struct {
	handler uintptr
	matNx   uint64
	matNy   uint64
}

func (tx *Tensor) NewMatFlatBuffer() (*MatFlatBuffer, error) {
	buf := &MatFlatBuffer{matNx: tx.nx, matNy: tx.ny}
	var code int
	mtNewMatFlatBuffer.Call(tx.handler,
		uintptr(unsafe.Pointer(&buf.handler)), uintptr(unsafe.Pointer(&code)))
	if mtCodeSuccess != code {
		return nil, &mtError{code, "failed to alloc"}
	}
	return buf, nil
}

func (buf *MatFlatBuffer) Destroy() {
	mtBufferDestroy.Call(buf.handler)
}

func (tx *Tensor) MatFlatMulVec(ty, tz *Tensor, buf *MatFlatBuffer, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != tz.nx ||
		tx.nx != buf.matNx || tx.ny != buf.matNy {
		return unsuitableShapeErr
	}
	var code int
	mtMatFlatMulVec.Call(tx.handler, ty.handler, tz.handler, buf.handler,
		uintptr(stream), uintptr(unsafe.Pointer(&code)))
	return newMtErr(code, "")
}
