package matrics

// ind

func (tx *Tensor) MatT(ty *Tensor, stream Stream) error {
	if tx.nx != ty.ny || tx.ny != ty.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatT(ty, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) MatAddScalar(ty *Tensor, a float32, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatAddScalar(ty, a, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) MatMulScalar(ty *Tensor, a float32, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatMulScalar(ty, a, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) MatMulAddScalar(ty *Tensor, a, b float32, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatMulAddScalar(ty, a, b, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) MatPowScalar(ty *Tensor, a float32, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatPowScalar(ty, a, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) MatPowMulScalar(ty *Tensor, a, b float32, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatPowMulScalar(ty, a, b, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) MatAddMat(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny || tx.nx != tz.nx || tx.ny != tz.ny {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatAddMat(ty, tz, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) MatSubMat(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != ty.ny || tx.nx != tz.nx || tx.ny != tz.ny {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatSubMat(ty, tz, stream, &code)
	return newMtErr(code, "")
}

// matmul

func (tx *Tensor) MatMulMat(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.ny || tx.ny != tz.ny || ty.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatMulMat(ty, tz, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) VecTMulMat(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.ny || ty.nx != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtVecTMulMat(ty, tz, stream, &code)
	return newMtErr(code, "")
}

func (tx *Tensor) MatMulVec(ty, tz *Tensor, stream Stream) error {
	if tx.nx != ty.nx || tx.ny != tz.nx {
		return unsuitableShapeErr
	}
	var code int
	tx.mtMatMulVec(ty, tz, stream, &code)
	return newMtErr(code, "")
}
