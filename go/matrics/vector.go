package matrics

// ind

func (tx *Tensor) VecAddScalar(ty *Tensor, a float32, stream Stream) error {
	if tx.nx != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecAddScalar(ty, a, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) VecMulScalar(ty *Tensor, a float32, stream Stream) error {
	if tx.nx != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecMulScalar(ty, a, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) VecMulAddScalar(ty *Tensor, a, b float32, stream Stream) error {
	if tx.nx != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecMulAddScalar(ty, a, b, stream, &code)
	return newMtErr(code, "")

}

func (tx *Tensor) VecPowScalar(ty *Tensor, a float32, stream Stream) error {
	if tx.nx != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecPowScalar(ty, a, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) VecPowMulScalar(ty *Tensor, a, b float32, stream Stream) error {
	if tx.nx != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecPowMulScalar(ty, a, b, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) VecAddVec(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecAddVec(ty, tz, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) VecSubVec(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecSubVec(ty, tz, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) VecPatchMulVec(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecPatchMulVec(ty, tz, stream, &code)
	return newMtErr(code, "")
}

// acc

func (tx *Tensor) VecSum(buf *Buffer, stream Stream) (float32, error) {
	if tx.nx != buf.ref {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	tx.mtVecSum(buf, stream, &res, &code)
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}

func (tx *Tensor) VecSquareSum(buf *Buffer, stream Stream) (float32, error) {
	if tx.nx != buf.ref {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	tx.mtVecSquareSum(buf, stream, &res, &code)
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}

func (tx *Tensor) VecMin(buf *Buffer, stream Stream) (float32, error) {
	if tx.nx != buf.ref {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	tx.mtVecMin(buf, stream, &res, &code)
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}

func (tx *Tensor) VecMax(buf *Buffer, stream Stream) (float32, error) {
	if tx.nx != buf.ref {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	tx.mtVecMax(buf, stream, &res, &code)
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}

func (tx *Tensor) VecDot(ty *Tensor, buf *Buffer, stream Stream) (float32, error) {
	if tx.nx != ty.nx || tx.nx != buf.ref {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	tx.mtVecDot(ty, buf, stream, &res, &code)
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}

func (tx *Tensor) VecSumSquareSum(ty *Tensor, buf *Buffer, stream Stream) (float32, error) {
	if tx.nx != ty.nx || tx.nx != buf.ref {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	tx.mtVecSumSquareSum(ty, buf, stream, &res, &code)
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}

func (tx *Tensor) VecDiffSquareSum(ty *Tensor, buf *Buffer, stream Stream) (float32, error) {
	if tx.nx != ty.nx || tx.nx != buf.ref {
		return 0, unsuitableShapeErr
	}
	var res float32
	var code int
	tx.mtVecDiffSquareSum(ty, buf, stream, &res, &code)
	if mtCodeSuccess != code {
		return 0, &mtError{code, ""}
	}
	return res, nil
}
